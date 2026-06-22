import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Decoupled Output Parameterization — per-sample row visualizations.

For each sample, saves a single tight row image:
  Ref Before | Ref After | Input X(x) | M·[X,1]^T | ||Δ(x)|| | Y(x)

Designed for manual curation: pick the best rows and stack them.
"""
import argparse
import json
import os
import random

import kornia
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from dataset import CompetitionDataset
from model import InRetouchNR, get_subpixel_sampling_windows


def forward_decomposed(model, x):
    B, C, H, W = x.shape
    x_ctx = x

    ctx_feat_all = model.local_context(x_ctx)
    _, cc, ch, cw = ctx_feat_all.shape
    sh, sw = (ch - H) // 2, (cw - W) // 2
    local_feat = ctx_feat_all[:, :, sh:sh+H, sw:sw+W]

    global_feat = model.global_context(x_ctx)
    global_feat = global_feat.view(-1, model.hidden_dim, 1, 1).expand(B, -1, H, W)

    dummy_off = torch.zeros(B, 2, H, W, device=x.device)
    rel_feat = model.rel_coord_pe(dummy_off)
    ys = torch.linspace(-1, 1, H, device=x.device)
    xs_coord = torch.linspace(-1, 1, W, device=x.device)
    gy, gx = torch.meshgrid(ys, xs_coord, indexing='ij')
    grid_abs = torch.stack([gx, gy], dim=-1).unsqueeze(0).permute(0, 3, 1, 2).expand(B, -1, -1, -1)
    abs_feat = model.abs_coord_pe(grid_abs, pixel_size=0.02)

    coord_feat = torch.cat([rel_feat, abs_feat], dim=1)
    color_feat = model.color_pe(x)

    mod_input = torch.cat([local_feat, global_feat], dim=1)
    mod_flat = mod_input.permute(0, 2, 3, 1).reshape(-1, model.cond_dim)
    film_params = model.film_gen(mod_flat)

    struct_feat = torch.cat([coord_feat, color_feat], dim=1)
    h = struct_feat.permute(0, 2, 3, 1).reshape(-1, model.mlp_in_dim)

    g1 = film_params[:, :model.hidden_dim*2]
    b1 = film_params[:, model.hidden_dim*2 : model.hidden_dim*4]
    h = model.mlp_layer1(h)
    h = h * (1.0 + g1) + b1
    h = F.silu(h)

    g2 = film_params[:, model.hidden_dim*4 : model.hidden_dim*5]
    b2 = film_params[:, model.hidden_dim*5 :]
    h = model.mlp_layer2(h)
    h = h * (1.0 + g2) + b2
    h = F.silu(h)

    head_out = model.mlp_head(h)
    matrix_flat = head_out[:, :12]
    detail_flat = head_out[:, 12:]

    M = matrix_flat.view(B, H, W, 3, 4)
    ones = torch.ones(B, 1, H, W, device=x.device)
    x_h = torch.cat([x, ones], dim=1).permute(0, 2, 3, 1).unsqueeze(-1)
    out_matrix = torch.matmul(M, x_h).squeeze(-1).permute(0, 3, 1, 2)

    detail_res = detail_flat.view(B, H, W, 3).permute(0, 3, 1, 2)
    residual = torch.tanh(detail_res) * 0.1

    final_out = out_matrix + residual

    return {
        'input': x,
        'matrix_M': M,
        'global_affine': out_matrix,
        'residual': residual,
        'final': torch.clamp(final_out, 0, 1),
    }


def run_tto(ref_in_tensor, ref_out_tensor, model, device, args):
    Hr, Wr = ref_in_tensor.shape[2:]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss(beta=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-4)

    cntx_pad = 14
    cntx_size = args.window_size + 2 * cntx_pad
    ws = args.window_size

    model.train()
    for step in range(args.steps):
        optimizer.zero_grad()
        pixel_h, pixel_w = 2.0 / Hr, 2.0 / Wr
        half_cntx_h = (cntx_size / 2.0) * pixel_h
        half_cntx_w = (cntx_size / 2.0) * pixel_w

        y_c = torch.rand(args.batch_size, device=device) * (2.0 - 2 * half_cntx_h) - (1.0 - half_cntx_h)
        x_c = torch.rand(args.batch_size, device=device) * (2.0 - 2 * half_cntx_w) - (1.0 - half_cntx_w)
        centers = torch.stack([x_c, y_c], dim=-1)

        patches_in_large_sharp, _, _, _, _ = get_subpixel_sampling_windows(
            ref_in_tensor, batch_size=args.batch_size, window_size=cntx_size, centers=centers)
        _, patches_in_smooth_13, _, _, _ = get_subpixel_sampling_windows(
            ref_in_tensor, batch_size=args.batch_size, window_size=ws, centers=centers)
        _, patches_target_smooth, offsets_target, abs_coords_target, _ = get_subpixel_sampling_windows(
            ref_out_tensor, batch_size=args.batch_size, window_size=ws, centers=centers)

        model.current_offsets = offsets_target.view(args.batch_size, 2, 1, 1).expand(-1, -1, ws, ws)
        model.current_abs_coords = abs_coords_target

        pred_windows = model(patches_in_smooth_13, x_ctx=patches_in_large_sharp, global_image=ref_in_tensor)

        loss_l1 = criterion(pred_windows, patches_target_smooth)
        pred_lab = kornia.color.rgb_to_lab(pred_windows)
        target_lab = kornia.color.rgb_to_lab(patches_target_smooth)
        loss_lab = torch.mean(torch.sqrt(torch.sum((pred_lab - target_lab) ** 2, dim=1) + 1e-8))
        ssim_val = kornia.metrics.ssim(pred_windows, patches_target_smooth, window_size=5)
        loss_ssim = 1.0 - ssim_val.mean()
        diff_h = torch.abs(pred_windows[:, :, 1:, :] - pred_windows[:, :, :-1, :])
        diff_w = torch.abs(pred_windows[:, :, :, 1:] - pred_windows[:, :, :, :-1])
        loss_tv = diff_h.mean() + diff_w.mean()

        loss = loss_l1 + 0.2 * loss_ssim + 0.05 * loss_lab + 0.001 * loss_tv
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 100 == 0 or step == 0:
            print(f"    Step {step+1:>4}/{args.steps}  loss={loss.item():.4f}")


def to_np(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def save_row(ref_b_np, ref_a_np, input_np, ga_np, res_mag, final_np, output_path):
    """
    Save a single tight row: Ref Before | Ref After | Input | Global Affine | Residual Mag | Output.
    No title, no labels, minimal margins.
    """
    images = [
        ('rgb', ref_b_np),
        ('rgb', ref_a_np),
        ('rgb', input_np),
        ('rgb', ga_np),
        ('heatmap', res_mag),
        ('rgb', final_np),
    ]

    n = len(images)
    # Use first image height as reference, scale all to same height
    target_h = input_np.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 3.2 * target_h / input_np.shape[1]))

    for i, (kind, img) in enumerate(images):
        ax = axes[i]
        if kind == 'rgb':
            ax.imshow(np.clip(img, 0, 1))
        else:
            ax.imshow(img, cmap='inferno')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.01, facecolor='white')
    plt.close(fig)


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    to_tensor = transforms.ToTensor()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CompetitionDataset(args.dataset_path)
    print(f"Dataset: {len(dataset)} samples")

    n = min(args.num_samples, len(dataset))
    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), n)
    print(f"Processing {n} random samples\n")

    stats_all = []

    for idx_i, sample_idx in enumerate(indices):
        task = dataset[sample_idx]
        sample_id = task['image_name']
        print(f"  [{idx_i+1:>2}/{n}] {sample_id}", end="  ", flush=True)

        input_img = Image.open(task['input_path']).convert('RGB')
        ref_in = Image.open(task['ref_input_path']).convert('RGB')
        ref_out = Image.open(task['ref_output_path']).convert('RGB')

        input_tensor = to_tensor(input_img).unsqueeze(0).to(device)
        ref_in_tensor = to_tensor(ref_in).unsqueeze(0).to(device)
        ref_out_tensor = to_tensor(ref_out).unsqueeze(0).to(device)

        # Build model + TTO
        model = InRetouchNR(hidden_dim=128).to(device)
        state_dict = torch.load(args.meta_weights, map_location=device)
        model.load_state_dict(state_dict)

        torch.manual_seed(args.seed + sample_idx)
        run_tto(ref_in_tensor, ref_out_tensor, model, device, args)

        # Decomposed inference
        model.eval()
        model.current_offsets = None
        model.current_abs_coords = None
        with torch.no_grad():
            comp = forward_decomposed(model, input_tensor)

        input_np = to_np(comp['input'].clamp(0, 1))
        ga_np = to_np(comp['global_affine'].clamp(0, 1))
        res_np = comp['residual'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        final_np = to_np(comp['final'])
        ref_b_np = to_np(ref_in_tensor.clamp(0, 1))
        ref_a_np = to_np(ref_out_tensor.clamp(0, 1))

        res_mag = np.sqrt(np.sum(res_np ** 2, axis=2))

        M_np = comp['matrix_M'].squeeze(0).cpu().numpy()
        M_avg = M_np.mean(axis=(0, 1))
        mean_res = float(np.abs(res_np).mean())
        ratio = mean_res / max(float(np.abs(ga_np).mean()), 1e-8)

        # Save row image
        out_path = os.path.join(args.output_dir, f'{sample_id}.png')
        save_row(ref_b_np, ref_a_np, input_np, ga_np, res_mag, final_np, out_path)

        stats_all.append({
            'sample_id': sample_id,
            'ratio': ratio,
            'mean_res': mean_res,
            'max_res': float(np.abs(res_np).max()),
        })

        print(f"ratio={ratio:.4f}  mean|Δ|={mean_res:.5f}  → {out_path}")

        del model
        torch.cuda.empty_cache()

    # Save stats JSON
    json_path = os.path.join(args.output_dir, 'stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats_all, f, indent=2)

    print(f"\nDone. {n} row images saved to {args.output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--meta_weights', type=str, default='weights/meta_model_ft.pth')
    parser.add_argument('--output_dir', type=str, default='decomposition_vis')
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=13)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=147)
    args = parser.parse_args()
    main(args)
