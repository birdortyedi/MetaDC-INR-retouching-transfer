import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Convergence Profiling: Meta-Learned Prior vs Random Initialization.

Runs test-time optimization (TTO) on N samples from the competition dataset
with two configurations:
  1. Starting from meta-learned weights (prior)
  2. Starting from random initialization (no prior)

At periodic checkpoints during TTO, performs full-image inference on the
reference "before" image and measures PSNR / SSIM against the reference
"after" image (ground truth for the reference pair).

Logs per-step metrics and generates a publication-quality comparison plot
with averaged convergence curves ± std across samples.
"""
import argparse
import json
import math
import os
import random

import kornia
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms

from dataset import CompetitionDataset
from model import InRetouchNR, get_subpixel_sampling_windows


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-10:
        return 50.0
    return 10 * math.log10(1.0 / mse)


def calculate_ssim_image(img1, img2):
    """Full-image SSIM using skimage (matches main.py evaluation)."""
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return ssim_metric(img1_np, img2_np, data_range=1.0, channel_axis=2)


def calculate_delta_e(pred, target):
    """Mean CIE ΔE (L*a*b*) between predicted and target."""
    pred_lab = kornia.color.rgb_to_lab(pred)
    target_lab = kornia.color.rgb_to_lab(target)
    de = torch.sqrt(torch.sum((pred_lab - target_lab) ** 2, dim=1) + 1e-8)
    return de.mean().item()


def run_tto_with_checkpoints(
    ref_in_tensor, ref_out_tensor, model, device, args,
    checkpoint_steps=None
):
    """
    Run TTO fitting the model to (ref_in → ref_out).
    At each checkpoint step, do full-image inference on ref_in and measure
    image-level PSNR/SSIM against ref_out.

    Returns:
        patch_metrics: dict of per-step patch-level metrics (loss, psnr, ssim, delta_e)
        image_metrics: dict of checkpoint-level full-image metrics (psnr, ssim, delta_e)
    """
    Hr, Wr = ref_in_tensor.shape[2:]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss(beta=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-4)

    cntx_pad = 14
    cntx_size = args.window_size + 2 * cntx_pad
    ws = args.window_size

    if checkpoint_steps is None:
        checkpoint_steps = set(range(0, args.steps, 10)) | {args.steps - 1}
    else:
        checkpoint_steps = set(checkpoint_steps)

    patch_metrics = {'step': [], 'loss': [], 'psnr': [], 'ssim': [], 'delta_e': []}
    image_metrics = {'step': [], 'psnr': [], 'ssim': [], 'delta_e': []}

    for step in range(args.steps):
        # --- Full-image checkpoint ---
        if step in checkpoint_steps:
            model.eval()
            model.current_offsets = None
            model.current_abs_coords = None
            with torch.no_grad():
                pred_full = model(ref_in_tensor).clamp(0, 1)
                img_psnr = calculate_psnr(pred_full, ref_out_tensor)
                img_ssim = calculate_ssim_image(pred_full, ref_out_tensor)
                img_de = calculate_delta_e(pred_full, ref_out_tensor)
            image_metrics['step'].append(step)
            image_metrics['psnr'].append(img_psnr)
            image_metrics['ssim'].append(img_ssim)
            image_metrics['delta_e'].append(img_de)

        # --- TTO step ---
        model.train()
        optimizer.zero_grad()

        pixel_h, pixel_w = 2.0 / Hr, 2.0 / Wr
        half_cntx_h = (cntx_size / 2.0) * pixel_h
        half_cntx_w = (cntx_size / 2.0) * pixel_w

        y_c = torch.rand(args.batch_size, device=device) * (2.0 - 2 * half_cntx_h) - (1.0 - half_cntx_h)
        x_c = torch.rand(args.batch_size, device=device) * (2.0 - 2 * half_cntx_w) - (1.0 - half_cntx_w)
        centers = torch.stack([x_c, y_c], dim=-1)

        patches_in_large_sharp, _, _, _, _ = get_subpixel_sampling_windows(
            ref_in_tensor, batch_size=args.batch_size, window_size=cntx_size, centers=centers
        )
        _, patches_in_smooth_13, _, _, _ = get_subpixel_sampling_windows(
            ref_in_tensor, batch_size=args.batch_size, window_size=ws, centers=centers
        )
        _, patches_target_smooth, offsets_target, abs_coords_target, _ = get_subpixel_sampling_windows(
            ref_out_tensor, batch_size=args.batch_size, window_size=ws, centers=centers
        )

        model.current_offsets = offsets_target.view(args.batch_size, 2, 1, 1).expand(-1, -1, ws, ws)
        model.current_abs_coords = abs_coords_target

        pred_windows = model(patches_in_smooth_13, x_ctx=patches_in_large_sharp, global_image=ref_in_tensor)

        # Losses (same as predict.py / main.py)
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

        # Log patch-level metrics every step
        with torch.no_grad():
            p_psnr = calculate_psnr(pred_windows, patches_target_smooth)
            p_ssim = ssim_val.mean().item()
            p_de = calculate_delta_e(pred_windows, patches_target_smooth)

        patch_metrics['step'].append(step)
        patch_metrics['loss'].append(loss.item())
        patch_metrics['psnr'].append(p_psnr)
        patch_metrics['ssim'].append(p_ssim)
        patch_metrics['delta_e'].append(p_de)

    # Final full-image checkpoint
    if (args.steps - 1) not in checkpoint_steps:
        model.eval()
        model.current_offsets = None
        model.current_abs_coords = None
        with torch.no_grad():
            pred_full = model(ref_in_tensor).clamp(0, 1)
            img_psnr = calculate_psnr(pred_full, ref_out_tensor)
            img_ssim = calculate_ssim_image(pred_full, ref_out_tensor)
            img_de = calculate_delta_e(pred_full, ref_out_tensor)
        image_metrics['step'].append(args.steps - 1)
        image_metrics['psnr'].append(img_psnr)
        image_metrics['ssim'].append(img_ssim)
        image_metrics['delta_e'].append(img_de)

    return patch_metrics, image_metrics


def average_metrics(all_metrics_list, key='step'):
    """Average metric curves across multiple samples. Returns mean ± std."""
    if not all_metrics_list:
        return {}

    # All runs should share the same steps
    steps = all_metrics_list[0][key]
    result = {key: steps}

    metric_keys = [k for k in all_metrics_list[0].keys() if k != key]
    for mk in metric_keys:
        stacked = np.array([m[mk] for m in all_metrics_list])
        result[f'{mk}_mean'] = stacked.mean(axis=0).tolist()
        result[f'{mk}_std'] = stacked.std(axis=0).tolist()

    return result


def plot_convergence(meta_img, rand_img, meta_patch, rand_patch, output_path, n_samples=1):
    """Generate a publication-quality 2×3 convergence comparison plot."""

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
    })

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(
        f'Convergence Profile: Meta-Learned Prior vs Random Init'
        f'  (averaged over {n_samples} sample{"s" if n_samples > 1 else ""})',
        fontsize=16, fontweight='bold', y=0.98
    )

    c_meta = '#2563EB'
    c_rand = '#DC2626'

    def _plot_panel(ax, steps_m, mean_m, std_m, steps_r, mean_r, std_r,
                    title, ylabel, higher_better):
        mean_m, std_m = np.array(mean_m), np.array(std_m)
        mean_r, std_r = np.array(mean_r), np.array(std_r)
        steps_m, steps_r = np.array(steps_m), np.array(steps_r)

        ax.plot(steps_m, mean_m, color=c_meta, lw=2.0, label='With Meta Prior', zorder=3)
        ax.plot(steps_r, mean_r, color=c_rand, lw=2.0, label='Random Init', ls='--', zorder=3)

        if n_samples > 1:
            ax.fill_between(steps_m, mean_m - std_m, mean_m + std_m,
                            alpha=0.15, color=c_meta)
            ax.fill_between(steps_r, mean_r - std_r, mean_r + std_r,
                            alpha=0.15, color=c_rand)
        else:
            # Fill gap between curves
            min_len = min(len(steps_m), len(steps_r))
            s = steps_m[:min_len]
            vm, vr = mean_m[:min_len], mean_r[:min_len]
            if higher_better:
                ax.fill_between(s, vr, vm, where=(vm >= vr), alpha=0.10, color=c_meta, interpolate=True)
            else:
                ax.fill_between(s, vm, vr, where=(vr >= vm), alpha=0.10, color=c_meta, interpolate=True)

        ax.set_title(title, fontweight='semibold')
        ax.set_xlabel('TTO Step')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', framealpha=0.9, edgecolor='#ccc')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

        # Annotate final values
        for vals, steps, color, yoff in [(mean_m, steps_m, c_meta, 10), (mean_r, steps_r, c_rand, -18)]:
            ax.annotate(f'{vals[-1]:.3f}', xy=(steps[-1], vals[-1]),
                        xytext=(-55, yoff), textcoords='offset points',
                        fontsize=9, color=color, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # Top row: Full-image metrics (the important ones)
    panels_top = [
        ('psnr', 'Full-Image PSNR (dB)', 'PSNR (dB)', True),
        ('ssim', 'Full-Image SSIM', 'SSIM', True),
        ('delta_e', 'Full-Image ΔE (CIE Lab)', 'ΔE', False),
    ]
    for ax, (key, title, ylabel, hb) in zip(axes[0], panels_top):
        _plot_panel(
            ax,
            meta_img['step'], meta_img[f'{key}_mean'], meta_img[f'{key}_std'],
            rand_img['step'], rand_img[f'{key}_mean'], rand_img[f'{key}_std'],
            title, ylabel, hb
        )

    # Bottom row: Patch-level metrics
    panels_bottom = [
        ('loss', 'Patch-Level Total Loss', 'Loss', False),
        ('psnr', 'Patch-Level PSNR (dB)', 'PSNR (dB)', True),
        ('ssim', 'Patch-Level SSIM', 'SSIM', True),
    ]
    for ax, (key, title, ylabel, hb) in zip(axes[1], panels_bottom):
        _plot_panel(
            ax,
            meta_patch['step'], meta_patch[f'{key}_mean'], meta_patch[f'{key}_std'],
            rand_patch['step'], rand_patch[f'{key}_mean'], rand_patch[f'{key}_std'],
            title, ylabel, hb
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    to_tensor = transforms.ToTensor()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = CompetitionDataset(args.dataset_path)
    print(f"Dataset: {len(dataset)} samples total")

    # Select N samples (random subset or first N)
    n = min(args.num_samples, len(dataset))
    if args.random_subset:
        random.seed(args.seed)
        indices = random.sample(range(len(dataset)), n)
    else:
        indices = list(range(n))

    print(f"Profiling {n} samples: indices {indices}")

    # Checkpoint schedule: dense early, sparse later
    checkpoint_steps = sorted(set(
        list(range(0, min(50, args.steps), 1)) +        # every step for first 50
        list(range(50, min(200, args.steps), 5)) +      # every 5 steps up to 200
        list(range(200, args.steps, 10)) +               # every 10 steps after 200
        [args.steps - 1]
    ))

    all_meta_patch, all_meta_image = [], []
    all_rand_patch, all_rand_image = [], []

    for idx_i, sample_idx in enumerate(indices):
        task = dataset[sample_idx]
        sample_id = task['image_name']
        print(f"\n{'='*60}")
        print(f"  Sample {idx_i+1}/{n}: {sample_id}")
        print(f"{'='*60}")

        ref_in = Image.open(task['ref_input_path']).convert('RGB')
        ref_out = Image.open(task['ref_output_path']).convert('RGB')
        ref_in_tensor = to_tensor(ref_in).unsqueeze(0).to(device)
        ref_out_tensor = to_tensor(ref_out).unsqueeze(0).to(device)

        print(f"  Ref size: {ref_in_tensor.shape[2:]}")

        # --- Run 1: With Meta Prior ---
        print(f"  [META] Running TTO with meta-learned prior...")
        model_meta = InRetouchNR(hidden_dim=args.hidden_dim).to(device)
        if os.path.exists(args.meta_weights):
            state_dict = torch.load(args.meta_weights, map_location=device)
            # Remove hidden_dim mismatch issues if any by strict=False or just ensure it matches
            model_meta.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Meta weights not found: {args.meta_weights}")

        torch.manual_seed(args.seed + sample_idx)
        pm, im = run_tto_with_checkpoints(
            ref_in_tensor, ref_out_tensor, model_meta, device, args, checkpoint_steps
        )
        all_meta_patch.append(pm)
        all_meta_image.append(im)
        del model_meta
        torch.cuda.empty_cache()

        # --- Run 2: Random Init ---
        print(f"  [RAND] Running TTO with random initialization...")
        model_rand = InRetouchNR(hidden_dim=args.hidden_dim).to(device)

        torch.manual_seed(args.seed + sample_idx)  # same seed for fair comparison
        pr, ir = run_tto_with_checkpoints(
            ref_in_tensor, ref_out_tensor, model_rand, device, args, checkpoint_steps
        )
        all_rand_patch.append(pr)
        all_rand_image.append(ir)
        del model_rand
        torch.cuda.empty_cache()

        # Per-sample summary
        print(f"  Final full-image PSNR:  Meta={im['psnr'][-1]:.2f} dB  |  Rand={ir['psnr'][-1]:.2f} dB")
        print(f"  Final full-image SSIM:  Meta={im['ssim'][-1]:.4f}     |  Rand={ir['ssim'][-1]:.4f}")

    # --- Aggregate across samples ---
    meta_patch_avg = average_metrics(all_meta_patch)
    rand_patch_avg = average_metrics(all_rand_patch)
    meta_image_avg = average_metrics(all_meta_image)
    rand_image_avg = average_metrics(all_rand_image)

    # --- Save raw data ---
    raw_data = {
        'meta_patch': meta_patch_avg,
        'rand_patch': rand_patch_avg,
        'meta_image': meta_image_avg,
        'rand_image': rand_image_avg,
        'config': vars(args),
        'sample_indices': indices,
    }
    json_path = os.path.join(args.output_dir, 'convergence_data.json')
    with open(json_path, 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nRaw metrics saved to {json_path}")

    # --- Generate plot ---
    plot_path = os.path.join(args.output_dir, 'convergence_plot.png')
    plot_convergence(
        meta_image_avg, rand_image_avg,
        meta_patch_avg, rand_patch_avg,
        plot_path, n_samples=n
    )

    # --- Print summary ---
    print(f"\n{'='*65}")
    print(f"  CONVERGENCE SUMMARY  (averaged over {n} samples)")
    print(f"{'='*65}")
    print(f"{'Metric':<22} {'Meta Prior':>12} {'Random Init':>12} {'Δ':>10}")
    print("-" * 58)
    for src, prefix in [(meta_image_avg, 'meta'), (rand_image_avg, 'rand')]:
        pass  # just for reference

    for key, label, fmt, higher in [
        ('psnr', 'Image PSNR (dB)', '.2f', True),
        ('ssim', 'Image SSIM', '.4f', True),
        ('delta_e', 'Image ΔE', '.2f', False),
    ]:
        vm = meta_image_avg[f'{key}_mean'][-1]
        vr = rand_image_avg[f'{key}_mean'][-1]
        diff = vm - vr
        sign = '+' if diff > 0 else ''
        print(f"  {label:<20} {vm:>12{fmt}} {vr:>12{fmt}} {sign}{diff:>9{fmt}}")

    print()
    for key, label, fmt, higher in [
        ('loss', 'Patch Loss', '.4f', False),
        ('psnr', 'Patch PSNR (dB)', '.2f', True),
        ('ssim', 'Patch SSIM', '.4f', True),
    ]:
        vm = meta_patch_avg[f'{key}_mean'][-1]
        vr = rand_patch_avg[f'{key}_mean'][-1]
        diff = vm - vr
        sign = '+' if diff > 0 else ''
        print(f"  {label:<20} {vm:>12{fmt}} {vr:>12{fmt}} {sign}{diff:>9{fmt}}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Profile convergence: meta-learned prior vs random init'
    )
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to competition dataset root (with sampleX/ dirs)')
    parser.add_argument('--meta_weights', type=str, default='weights/meta_model_ft.pth',
                        help='Path to meta-learned weights checkpoint')
    parser.add_argument('--output_dir', type=str, default='convergence_profiles',
                        help='Directory to save plots and metric logs')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to profile (averaged for curves)')
    parser.add_argument('--random_subset', action='store_true',
                        help='Randomly pick samples instead of first N')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of TTO steps')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for patch sampling')
    parser.add_argument('--window_size', type=int, default=13)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension of the MLP')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=147,
                        help='Random seed for reproducible patch sampling')
    args = parser.parse_args()
    main(args)
