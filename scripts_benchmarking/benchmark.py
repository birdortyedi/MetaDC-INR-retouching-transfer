import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import thop
import lpips
from skimage.metrics import structural_similarity as ssim_metric
from model import InRetouchNR

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def calculate_ssim(img1, img2):
    img1_np = img1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return ssim_metric(img1_np, img2_np, data_range=1.0, channel_axis=2)

import kornia
from model import get_subpixel_sampling_windows

def run_tto(model_path, target_natural, ref_natural, ref_retouched, device, steps=500, batch_size=512, window_size=13, hidden_dim=128):
    model = InRetouchNR(hidden_dim=hidden_dim).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load images
    to_tensor = transforms.ToTensor()
    t_nat = to_tensor(Image.open(target_natural).convert('RGB')).unsqueeze(0).to(device)
    r_nat = to_tensor(Image.open(ref_natural).convert('RGB')).unsqueeze(0).to(device)
    r_ret = to_tensor(Image.open(ref_retouched).convert('RGB')).unsqueeze(0).to(device)
    
    Hr, Wr = r_nat.shape[2:]
    cntx_size = window_size + 28
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.SmoothL1Loss(beta=0.01)
    
    # TTO Loop
    torch.cuda.synchronize()
    tto_start = time.time()
    
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        
        # Exact Sampling Logic from predict.py
        pixel_h, pixel_w = 2.0 / Hr, 2.0 / Wr
        half_cntx_h = (cntx_size / 2.0) * pixel_h
        half_cntx_w = (cntx_size / 2.0) * pixel_w
        
        y_c = torch.rand(batch_size, device=device) * (2.0 - 2*half_cntx_h) - (1.0 - half_cntx_h)
        x_c = torch.rand(batch_size, device=device) * (2.0 - 2*half_cntx_w) - (1.0 - half_cntx_w)
        centers = torch.stack([x_c, y_c], dim=-1)

        p_large_sharp, _, _, _, _ = get_subpixel_sampling_windows(r_nat, batch_size=batch_size, window_size=cntx_size, centers=centers)
        _, p_smooth_13, _, _, _ = get_subpixel_sampling_windows(r_nat, batch_size=batch_size, window_size=window_size, centers=centers)
        _, p_target_smooth, offsets_target, abs_coords_target, _ = get_subpixel_sampling_windows(r_ret, batch_size=batch_size, window_size=window_size, centers=centers)
        
        model.current_offsets = offsets_target.view(batch_size, 2, 1, 1).expand(-1, -1, window_size, window_size)
        model.current_abs_coords = abs_coords_target
        
        pred = model(p_smooth_13, x_ctx=p_large_sharp, global_image=r_nat)
        
        # Losses
        loss_l1 = criterion(pred, p_target_smooth)
        pred_lab = kornia.color.rgb_to_lab(pred)
        target_lab = kornia.color.rgb_to_lab(p_target_smooth)
        loss_lab = torch.mean(torch.sqrt(torch.sum((pred_lab - target_lab)**2, dim=1) + 1e-8))
        loss_ssim = 1.0 - kornia.metrics.ssim(pred, p_target_smooth, window_size=5).mean()
        
        loss = loss_l1 + 0.2 * loss_ssim + 0.05 * loss_lab
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    tto_end = time.time()
    
    # Final Inference
    model.eval()
    model.current_offsets = None
    model.current_abs_coords = None
    with torch.no_grad():
        output = model(t_nat).clamp(0, 1)
    
    return output, (tto_end - tto_start)

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    if args.skip_accuracy:
        print("Accuracy benchmark skipped.")
        return

    # LPIPS
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # Dataset Setup
    dataset_root = args.dataset_path
    bench_root = os.path.join(dataset_root, "Benchmark")
    ref_file = os.path.join(bench_root, "references_file.txt")
    
    with open(ref_file, 'r') as f:
        pairs = [line.strip().split(',') for line in f.readlines()]
    
    presets = [f"Preset_{i}" for i in range(146, 168)]
    
    if args.num_samples > 0:
        pairs = pairs[:args.num_samples]
    # Default is 0, which means all pairs
        
    if args.num_presets > 0:
        presets = presets[:args.num_presets]
    # Default is 0, which means all presets
        
    steps_to_test = [10, 100, 500]
    results_summary = []

    print(f"\nStarting Comparative TTO Benchmark ({len(pairs)} pairs x {len(presets)} presets)")
    print(f"Total evaluations per step count: {len(pairs) * len(presets)}")
    print(f"Steps to evaluate: {steps_to_test}")

    for steps in steps_to_test:
        print(f"\n>>> Evaluating TTO with {steps} steps...")
        psnr_vals, ssim_vals, lpips_vals, time_vals = [], [], [], []
        
        pbar = tqdm(total=len(pairs) * len(presets), desc=f"Steps {steps}")
        for target_name, ref_name in pairs:
            for preset in presets:
                t_nat_path = os.path.join(bench_root, "Test", "natural", target_name)
                r_nat_path = os.path.join(bench_root, "Test_References", "natural", ref_name)
                r_ret_path = os.path.join(bench_root, "Test_References", "Presets", preset, ref_name)
                gt_path = os.path.join(bench_root, "Test", "Presets", preset, target_name)
                
                if not all(os.path.exists(p) for p in [t_nat_path, r_nat_path, r_ret_path, gt_path]):
                    pbar.update(1)
                    continue
                    
                output, t_time = run_tto(args.meta_weights, t_nat_path, r_nat_path, r_ret_path, device, 
                                         steps=steps, batch_size=args.batch_size, window_size=args.window_size,
                                         hidden_dim=args.hidden_dim)
                
                gt = transforms.ToTensor()(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
                if output.shape != gt.shape:
                    output = F.interpolate(output, size=gt.shape[2:], mode='bilinear', align_corners=False)
                
                psnr_vals.append(calculate_psnr(output, gt))
                ssim_vals.append(calculate_ssim(output, gt))
                lpips_vals.append(loss_fn_alex(output * 2 - 1, gt * 2 - 1).item())
                time_vals.append(t_time)
                
                pbar.update(1)
                pbar.set_description(f"PSNR: {np.mean(psnr_vals):.2f}")
        
        results_summary.append({
            'steps': steps,
            'psnr': np.mean(psnr_vals),
            'ssim': np.mean(ssim_vals),
            'lpips': np.mean(lpips_vals),
            'time': np.mean(time_vals)
        })

    # Print Comparison Table
    print("\n" + "="*70)
    print("      COMPARATIVE TTO PERFORMANCE SUMMARY (RTD)")
    print("="*70)
    print(f"{'Steps':<10} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'LPIPS':<10} | {'Time (s)':<10}")
    print("-" * 70)
    for res in results_summary:
        print(f"{res['steps']:<10} | {res['psnr']:>10.2f} | {res['ssim']:>8.4f} | {res['lpips']:>8.4f} | {res['time']:>8.2f}")
    print("="*70 + "\n")

    # Final report to file
    with open("FINAL_BENCHMARK_RESULTS_multiscale.txt", "w") as f:
        f.write("="*70 + "\n")
        f.write("      META-DC INR ACCURACY BENCHMARK RESULTS (RTD)\n")
        f.write("="*70 + "\n\n")
        
        f.write("COMPARATIVE TTO PERFORMANCE\n")
        f.write(f"{'Steps':<10} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'LPIPS':<10} | {'Time (s)':<10}\n")
        f.write("-" * 70 + "\n")
        for res in results_summary:
            f.write(f"{res['steps']:<10} | {res['psnr']:>10.2f} | {res['ssim']:>8.4f} | {res['lpips']:>8.4f} | {res['time']:>8.2f}\n")
        f.write("\n" + "="*70 + "\n")
    
    print(f"\nFinal results saved to FINAL_BENCHMARK_RESULTS.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--meta_weights', type=str, default='weights/meta_model_ft.pth')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples (0=all)')
    parser.add_argument('--num_presets', type=int, default=0, help='Number of presets (0=all)')
    parser.add_argument('--skip_accuracy', action='store_true', help='Only profile complexity')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--window_size', type=int, default=13, help='Window size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    args = parser.parse_args()
    main(args)
