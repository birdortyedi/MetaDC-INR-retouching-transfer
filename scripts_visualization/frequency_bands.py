import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Frequency Band Error — Final figure for samples 151 & 157.

Pixel-perfect stitching with PIL. No matplotlib margins.
Each row: Input | GT | Low-freq err (w/o Δ) | Low-freq err (w/ Δ)
Two rows stacked. 2-3px gap between panels, smallest edge equalized.
"""
import argparse
import os

import kornia
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from scipy.ndimage import gaussian_filter
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
        'global_affine': out_matrix,
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


def bandpass_filter(img_np, f_low, f_high):
    H, W, C = img_np.shape
    cy, cx = H // 2, W // 2
    y_grid, x_grid = np.ogrid[-cy:H-cy, -cx:W-cx]
    r = np.sqrt((y_grid / H) ** 2 + (x_grid / W) ** 2)
    mask = ((r >= f_low) & (r < f_high)).astype(np.float64)
    mask = gaussian_filter(mask, sigma=1.0)
    filtered = np.zeros_like(img_np)
    for c in range(C):
        F_ch = np.fft.fftshift(np.fft.fft2(img_np[:, :, c]))
        filtered[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(F_ch * mask)))
    return filtered


def error_magnitude(err_rgb):
    return np.sqrt(np.sum(err_rgb ** 2, axis=2))


def heatmap_to_rgb(mag, vmax, cmap_name='inferno'):
    """Convert [H,W] magnitude to [H,W,3] uint8 RGB via colormap."""
    normed = np.clip(mag / max(vmax, 1e-8), 0, 1)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(normed)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def float_to_uint8(img_np):
    return (np.clip(img_np, 0, 1) * 255).astype(np.uint8)


def resize_to_height(pil_img, target_h):
    """Resize preserving aspect ratio so height == target_h."""
    w, h = pil_img.size
    new_w = int(round(w * target_h / h))
    return pil_img.resize((new_w, target_h), Image.LANCZOS)


def stitch_row(panels_uint8, gap=3):
    """
    Stitch a list of [H,W,3] uint8 arrays into one row with `gap` px between.
    All panels must have the same height.
    """
    h = panels_uint8[0].shape[0]
    total_w = sum(p.shape[1] for p in panels_uint8) + gap * (len(panels_uint8) - 1)
    canvas = np.ones((h, total_w, 3), dtype=np.uint8) * 255  # white gap
    x = 0
    for p in panels_uint8:
        canvas[:, x:x+p.shape[1]] = p
        x += p.shape[1] + gap
    return canvas


def stitch_rows_vertical(rows, gap=3):
    """Stack row images vertically with `gap` px between."""
    max_w = max(r.shape[1] for r in rows)
    # Pad narrower rows to max width (white)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.ones((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8) * 255
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    total_h = sum(r.shape[0] for r in padded) + gap * (len(padded) - 1)
    canvas = np.ones((total_h, max_w, 3), dtype=np.uint8) * 255
    y = 0
    for r in padded:
        canvas[y:y+r.shape[0]] = r
        y += r.shape[0] + gap
    return canvas


def process_sample(dataset, sample_name, device, args):
    """Run TTO + decomposition for a named sample. Returns panels as uint8."""
    to_tensor = transforms.ToTensor()

    # Find sample by name
    idx = None
    for i in range(len(dataset)):
        if dataset[i]['image_name'] == sample_name:
            idx = i
            break
    if idx is None:
        raise ValueError(f"Sample '{sample_name}' not found in dataset")

    task = dataset[idx]
    print(f"\n  Processing {sample_name}...")

    ref_in = Image.open(task['ref_input_path']).convert('RGB')
    ref_out = Image.open(task['ref_output_path']).convert('RGB')

    ref_in_tensor = to_tensor(ref_in).unsqueeze(0).to(device)
    ref_out_tensor = to_tensor(ref_out).unsqueeze(0).to(device)

    model = InRetouchNR(hidden_dim=128).to(device)
    state_dict = torch.load(args.meta_weights, map_location=device)
    model.load_state_dict(state_dict)

    torch.manual_seed(args.seed)
    run_tto(ref_in_tensor, ref_out_tensor, model, device, args)

    model.eval()
    model.current_offsets = None
    model.current_abs_coords = None
    with torch.no_grad():
        comp = forward_decomposed(model, ref_in_tensor)

    ref_in_np = to_np(ref_in_tensor.clamp(0, 1))
    affine_np = to_np(comp['global_affine'].clamp(0, 1))
    final_np = to_np(comp['final'])

    # Resize GT to match model output
    H_out, W_out = affine_np.shape[:2]
    gt_resized = F.interpolate(ref_out_tensor, size=(H_out, W_out),
                               mode='bilinear', align_corners=False)
    gt_np = to_np(gt_resized.clamp(0, 1))

    # Low-freq band error
    f_lo, f_hi = 0.0, 0.10
    gt_low = bandpass_filter(gt_np, f_lo, f_hi)
    aff_low = bandpass_filter(affine_np, f_lo, f_hi)
    fin_low = bandpass_filter(final_np, f_lo, f_hi)

    err_aff = error_magnitude(aff_low - gt_low)
    err_fin = error_magnitude(fin_low - gt_low)
    vmax = max(err_aff.max(), err_fin.max(), 1e-8)

    mae_aff = err_aff.mean()
    mae_fin = err_fin.mean()
    red = (mae_aff - mae_fin) / max(mae_aff, 1e-8) * 100
    print(f"    Error reduction: {red:.1f}%  (MAE aff={mae_aff:.5f} → fin={mae_fin:.5f})")

    panels = {
        'sample_name': sample_name,
        'input': float_to_uint8(ref_in_np),
        'gt': float_to_uint8(gt_np),
        'err_aff': heatmap_to_rgb(err_aff, vmax),
        'err_fin': heatmap_to_rgb(err_fin, vmax),
        'err_aff_raw': err_aff,   # float [H,W] for shared colorbar
        'err_fin_raw': err_fin,   # float [H,W] for shared colorbar
    }

    del model
    torch.cuda.empty_cache()

    return panels


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = CompetitionDataset(args.dataset_path)
    print(f"Dataset: {len(dataset)} samples")

    sample_names = [s.strip() for s in args.samples.split(',')]
    all_data = []

    for name in sample_names:
        panels = process_sample(dataset, name, device, args)
        all_data.append(panels)

    # ─── Save individual panels for LaTeX ───
    # Compute global vmax for shared scale across all samples
    global_vmax = 0
    for panels in all_data:
        global_vmax = max(global_vmax, panels['err_aff_raw'].max(), panels['err_fin_raw'].max())
    
    print(f"\nGlobal vmax for heatmaps: {global_vmax:.5f}")

    for panels in all_data:
        name = panels['sample_name']
        
        # Save Before (Input)
        input_img = Image.fromarray(panels['input'])
        input_img.save(os.path.join(args.output_dir, f"{name}_before.png"))
        
        # Save After (GT)
        gt_img = Image.fromarray(panels['gt'])
        gt_img.save(os.path.join(args.output_dir, f"{name}_after.png"))
        
        # Save Error Affine (shared scale)
        err_aff_rgb = heatmap_to_rgb(panels['err_aff_raw'], global_vmax)
        err_aff_img = Image.fromarray(err_aff_rgb)
        err_aff_img.save(os.path.join(args.output_dir, f"{name}_error_affine.png"))
        
        # Save Error Full (shared scale)
        err_fin_rgb = heatmap_to_rgb(panels['err_fin_raw'], global_vmax)
        err_fin_img = Image.fromarray(err_fin_rgb)
        err_fin_img.save(os.path.join(args.output_dir, f"{name}_error_full.png"))

    print(f"\nAll individual panels saved to {args.output_dir}")

    # ─── Generate Combined Single Image (consistent with failure analysis) ───
    from PIL import ImageDraw, ImageFont
    serif_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf"
    ]
    font_size = 48
    font_header = None
    for fp in serif_paths:
        if os.path.exists(fp):
            font_header = ImageFont.truetype(fp, font_size)
            break
    if font_header is None: font_header = ImageFont.load_default()

    labels = ["Ref. Before", "Ref. After", "Low-Freq. Error (w/o Δ(x))", "Low-Freq. Error (w/ Δ(x))"]
    rows = []
    spacing = 3
    header_margin = 100
    
    for panels in all_data:
        # Before, After, Err-Aff, Err-Fin
        # Use shared global_vmax for consistent comparison
        err_aff_rgb = heatmap_to_rgb(panels['err_aff_raw'], global_vmax)
        err_fin_rgb = heatmap_to_rgb(panels['err_fin_raw'], global_vmax)
        
        raw_imgs = [panels['input'], panels['gt'], err_aff_rgb, err_fin_rgb]
        pil_panels = [Image.fromarray(img) for img in raw_imgs]
        
        h = pil_panels[0].height
        resized = []
        for p in pil_panels:
            w = int(p.width * h / p.height)
            resized.append(p.resize((w, h), Image.LANCZOS))
            
        row_w = sum(p.width for p in resized) + spacing * (len(resized) - 1)
        row_canvas = Image.new('RGB', (row_w, h), (255, 255, 255))
        curr_x = 0
        panel_x_starts = []
        for p in resized:
            panel_x_starts.append(curr_x)
            row_canvas.paste(p, (curr_x, 0))
            curr_x += p.width + spacing
        rows.append({'img': row_canvas, 'x_starts': panel_x_starts, 'panel_widths': [p.width for p in resized]})

    if rows:
        max_w = max(r['img'].width for r in rows)
        total_h = sum(r['img'].height for r in rows) + spacing * (len(rows) - 1) + header_margin
        
        final_canvas = Image.new('RGB', (max_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(final_canvas)
        
        # Draw Headers
        first_row = rows[0]
        for x, w, label in zip(first_row['x_starts'], first_row['panel_widths'], labels):
            bbox = draw.textbbox((0, 0), label, font=font_header)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx = x + (w - tw) // 2
            ly = (header_margin - th) // 2 - bbox[1]
            draw.text((lx, ly), label, fill=(0, 0, 0), font=font_header)
            
        curr_y = header_margin
        for r in rows:
            final_canvas.paste(r['img'], (0, curr_y))
            curr_y += r['img'].height + spacing
            
        combined_path = os.path.join(args.output_dir, "frequency_bands_combined.png")
        final_canvas.save(combined_path, quality=95)
        print(f"\nCombined frequency analysis saved to {combined_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--meta_weights', type=str, default='weights/meta_model_ft.pth')
    parser.add_argument('--output_dir', type=str, default='frequency_bands')
    parser.add_argument('--samples', type=str, default='sample151,sample157',
                        help='Comma-separated sample names')
    parser.add_argument('--gap', type=int, default=3)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=13)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=147)
    args = parser.parse_args()
    main(args)
