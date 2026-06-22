import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms
from tqdm import tqdm
import kornia
import lpips

from dataset import RetouchEvaluationDataset
from model import InRetouchNR, get_subpixel_sampling_windows


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return float(ssim_metric(img1_np, img2_np, data_range=1.0, channel_axis=2))


def run_tto_task(task, model_path, device, seed, batch_size=484, window_size=13, lr=1e-3):
    # Set seed for reproducibility of TTO sampling on this specific task/seed combination
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    to_tensor = transforms.ToTensor()

    input_img = Image.open(task['input_path']).convert('RGB')
    ref_in = Image.open(task['ref_input_path']).convert('RGB')
    ref_out = Image.open(task['ref_output_path']).convert('RGB')
    gt_img = Image.open(task['gt_path']).convert('RGB')

    input_tensor = to_tensor(input_img).unsqueeze(0).to(device)
    ref_in_tensor = to_tensor(ref_in).unsqueeze(0).to(device)
    ref_out_tensor = to_tensor(ref_out).unsqueeze(0).to(device)
    gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device)

    Hr, Wr = ref_in_tensor.shape[2:]

    # Initialize model and load weights
    model = InRetouchNR(hidden_dim=128).to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Weights not found at: {model_path}")

    # Freeze CNN feature extractors
    for param in model.local_context.parameters():
        param.requires_grad = False
    for param in model.global_context.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.SmoothL1Loss(beta=0.01)

    # 100 steps total
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

    cntx_pad = 14
    cntx_size = window_size + 2 * cntx_pad
    ws = window_size

    metrics = {}

    # Optimization Loop
    for step in range(100):
        # Step 10 Evaluation
        if step == 10:
            model.eval()
            model.current_offsets = None
            model.current_abs_coords = None
            with torch.no_grad():
                pred_10 = model(input_tensor).clamp(0, 1)
                if pred_10.shape != gt_tensor.shape:
                    pred_10 = nn.functional.interpolate(pred_10, size=gt_tensor.shape[2:], mode='bilinear', align_corners=False)
                
                metrics[10] = {
                    'psnr': float(calculate_psnr(pred_10, gt_tensor)),
                    'ssim': float(calculate_ssim(pred_10, gt_tensor)),
                    'pred_tensor': pred_10.cpu() # saved temporarily to compute LPIPS on GPU or CPU later
                }

        model.train()
        optimizer.zero_grad()

        pixel_h, pixel_w = 2.0 / Hr, 2.0 / Wr
        half_cntx_h, half_cntx_w = (cntx_size / 2.0) * pixel_h, (cntx_size / 2.0) * pixel_w

        y_c = torch.rand(batch_size, device=device) * (2.0 - 2 * half_cntx_h) - (1.0 - half_cntx_h)
        x_c = torch.rand(batch_size, device=device) * (2.0 - 2 * half_cntx_w) - (1.0 - half_cntx_w)
        centers = torch.stack([x_c, y_c], dim=-1)

        # Subpixel Samples
        patches_in_large_sharp, _, _, _, _ = get_subpixel_sampling_windows(
            ref_in_tensor, batch_size=batch_size, window_size=cntx_size, centers=centers
        )
        _, patches_in_smooth_13, _, _, _ = get_subpixel_sampling_windows(
            ref_in_tensor, batch_size=batch_size, window_size=ws, centers=centers
        )
        _, patches_target_smooth, offsets_target, abs_coords_target, _ = get_subpixel_sampling_windows(
            ref_out_tensor, batch_size=batch_size, window_size=ws, centers=centers
        )

        model.current_offsets = offsets_target.view(batch_size, 2, 1, 1).expand(-1, -1, ws, ws)
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

    # Step 100 Evaluation
    model.eval()
    model.current_offsets = None
    model.current_abs_coords = None
    with torch.no_grad():
        pred_100 = model(input_tensor).clamp(0, 1)
        if pred_100.shape != gt_tensor.shape:
            pred_100 = nn.functional.interpolate(pred_100, size=gt_tensor.shape[2:], mode='bilinear', align_corners=False)

        metrics[100] = {
            'psnr': float(calculate_psnr(pred_100, gt_tensor)),
            'ssim': float(calculate_ssim(pred_100, gt_tensor)),
            'pred_tensor': pred_100.cpu()
        }

    return metrics, gt_tensor.cpu()


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # LPIPS Alex
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    dataset = RetouchEvaluationDataset(args.dataset_path)
    print(f"Found {len(dataset)} benchmark tasks.")

    seeds = [42, 100, 2026, 999, 12345]
    os.makedirs(args.output_dir, exist_ok=True)
    temp_json_path = os.path.join(args.output_dir, "variance_results_temp.json")
    final_json_path = os.path.join(args.output_dir, "variance_results.json")
    final_txt_path = os.path.join(args.output_dir, "variance_results.txt")

    # Load progress if exists
    progress = {}
    if os.path.exists(temp_json_path):
        try:
            with open(temp_json_path, 'r') as f:
                progress = json.load(f)
            print(f"Loaded existing progress. {len(progress)} tasks already processed.")
        except Exception as e:
            print(f"Failed to load progress file: {e}. Starting fresh.")

    # We iterate over tasks first, then seeds, so we can save incrementally per task.
    for idx in range(len(dataset)):
        task = dataset[idx]
        task_key = str(idx)

        # Check if task is already processed for all 5 seeds
        if task_key in progress and len(progress[task_key].get("seeds", {})) == len(seeds):
            continue

        if task_key not in progress:
            progress[task_key] = {
                "preset": task['preset'],
                "image_name": task['image_name'],
                "seeds": {}
            }

        print(f"\nProcessing Task {idx+1}/{len(dataset)} | Preset: {task['preset']} | Image: {task['image_name']}")

        for seed in seeds:
            seed_key = str(seed)
            if seed_key in progress[task_key]["seeds"]:
                continue

            start_t = time.time()
            try:
                metrics, gt_cpu = run_tto_task(
                    task, args.meta_weights, device, seed,
                    batch_size=args.batch_size, window_size=args.window_size, lr=args.lr
                )
                
                # Compute LPIPS on GPU
                gt_gpu = gt_cpu.to(device)
                
                lpips_10 = float(loss_fn_alex(metrics[10]['pred_tensor'].to(device) * 2 - 1, gt_gpu * 2 - 1).item())
                lpips_100 = float(loss_fn_alex(metrics[100]['pred_tensor'].to(device) * 2 - 1, gt_gpu * 2 - 1).item())
                
                progress[task_key]["seeds"][seed_key] = {
                    "10": {
                        "psnr": float(metrics[10]['psnr']),
                        "ssim": float(metrics[10]['ssim']),
                        "lpips": lpips_10
                    },
                    "100": {
                        "psnr": float(metrics[100]['psnr']),
                        "ssim": float(metrics[100]['ssim']),
                        "lpips": lpips_100
                    }
                }
                
                print(f"  Seed {seed:<5} | 10 steps: PSNR {metrics[10]['psnr']:.2f} dB, LPIPS {lpips_10:.4f} | 100 steps: PSNR {metrics[100]['psnr']:.2f} dB, LPIPS {lpips_100:.4f} | Time: {time.time()-start_t:.1f}s")
            except Exception as e:
                print(f"  Error on Task {idx+1} Seed {seed}: {e}")
                continue

        # Save progress after each task
        with open(temp_json_path, 'w') as f:
            json.dump(progress, f, indent=2)

    completed_task_keys = [k for k, v in progress.items() if len(v["seeds"]) == len(seeds)]
    if not completed_task_keys:
        print("No completed tasks to generate statistics.")
        return

    print(f"\nComputing statistics over {len(completed_task_keys)} completed tasks...")

    # Initialize container for overall scores per seed
    seed_scores = {
        seed: {
            "10": {"psnr": [], "ssim": [], "lpips": []},
            "100": {"psnr": [], "ssim": [], "lpips": []}
        } for seed in seeds
    }

    for task_key in completed_task_keys:
        task_data = progress[task_key]
        for seed in seeds:
            seed_key = str(seed)
            for step in ["10", "100"]:
                metrics = task_data["seeds"][seed_key][step]
                seed_scores[seed][step]["psnr"].append(metrics["psnr"])
                seed_scores[seed][step]["ssim"].append(metrics["ssim"])
                seed_scores[seed][step]["lpips"].append(metrics["lpips"])

    # Average over all tasks per seed
    overall_per_seed = {
        step: {
            metric: [float(np.mean(seed_scores[seed][step][metric])) for seed in seeds]
            for metric in ["psnr", "ssim", "lpips"]
        } for step in ["10", "100"]
    }

    lines = []
    lines.append("==========================================================================")
    lines.append("                  METADC-INR TTO VARIANCE ANALYSIS")
    lines.append("==========================================================================")
    lines.append(f"Evaluated on {len(completed_task_keys)} benchmark tasks across 5 seeds: {seeds}")
    lines.append("==========================================================================\n")

    lines.append("### 1. Overall Performance Per Seed")
    lines.append(f"| Seed | 10-step PSNR | 10-step SSIM | 10-step LPIPS | 100-step PSNR | 100-step SSIM | 100-step LPIPS |")
    lines.append(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for i, seed in enumerate(seeds):
        p10 = overall_per_seed["10"]["psnr"][i]
        s10 = overall_per_seed["10"]["ssim"][i]
        l10 = overall_per_seed["10"]["lpips"][i]
        p100 = overall_per_seed["100"]["psnr"][i]
        s100 = overall_per_seed["100"]["ssim"][i]
        l100 = overall_per_seed["100"]["lpips"][i]
        lines.append(f"| {seed} | {p10:.4f} | {s10:.4f} | {l10:.4f} | {p100:.4f} | {s100:.4f} | {l100:.4f} |")
    lines.append("")

    lines.append("### 2. Summary Statistics (Across 5 Seeds)")
    lines.append(f"| Metric | 10-step adapt | 100-step adapt |")
    lines.append(f"| :--- | :---: | :---: |")
    for metric in ["psnr", "ssim", "lpips"]:
        vals_10 = overall_per_seed["10"][metric]
        vals_100 = overall_per_seed["100"][metric]

        mean_10, std_10 = np.mean(vals_10), np.std(vals_10)
        mean_100, std_100 = np.mean(vals_100), np.std(vals_100)

        fmt = ".4f" if metric != "psnr" else ".2f"
        lines.append(f"| {metric.upper()} (Mean ± Std) | {mean_10:{fmt}} ± {std_10:{fmt}} | {mean_100:{fmt}} ± {std_100:{fmt}} |")
        lines.append(f"| {metric.upper()} (Min / Max) | {np.min(vals_10):{fmt}} / {np.max(vals_10):{fmt}} | {np.min(vals_100):{fmt}} / {np.max(vals_100):{fmt}} |")

    report_content = "\n".join(lines)
    print("\n" + report_content + "\n")

    with open(final_txt_path, 'w') as f:
        f.write(report_content)

    with open(final_json_path, 'w') as f:
        json.dump({
            "overall_per_seed": overall_per_seed,
            "raw_progress": progress
        }, f, indent=2)

    print(f"Results saved to:\n  - {final_txt_path}\n  - {final_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to Retouch_Transfer_Dataset")
    parser.add_argument('--meta_weights', type=str, default='weights/meta_model_ft.pth', help="Path to meta-model weights")
    parser.add_argument('--output_dir', type=str, default='logs_and_reports', help="Output directory")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--batch_size', type=int, default=484, help="Batch size for TTO")
    parser.add_argument('--window_size', type=int, default=13, help="INR window size")
    parser.add_argument('--lr', type=float, default=1e-3, help="TTO learning rate")
    args = parser.parse_args()
    main(args)
