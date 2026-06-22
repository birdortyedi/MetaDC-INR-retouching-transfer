import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import lpips
import kornia

from tto_utils import run_tto_standard, run_tto_lie

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set default weights based on model type if not provided
    if args.meta_weights is None:
        if args.model_type == 'standard':
            args.meta_weights = 'weights/meta_model_ft.pth'
        else:
            args.meta_weights = 'weights/meta_model_lie_h128_final.pth'
            
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Using Weights: {args.meta_weights}")
    
    bench_root = os.path.join(args.dataset_path, "Benchmark")
    ref_file = os.path.join(bench_root, "references_file.txt")
    
    with open(ref_file, 'r') as f:
        pairs = [line.strip().split(',') for line in f.readlines() if line.strip()]
        
    presets = [f"Preset_{i}" for i in range(146, 168)]
    
    if args.num_samples > 0:
        pairs = pairs[:args.num_samples]
    if args.num_presets > 0:
        presets = presets[:args.num_presets]

    strategies = ['full', 'selective', 'head_only']
    results = {s: [] for s in strategies}

    print(f"\nEvaluating TTO Strategies on {len(pairs)} pairs x {len(presets)} presets = {len(pairs)*len(presets)} combinations.")
    print(f"Number of TTO steps: {args.steps}")
    
    for target_name, ref_name in pairs:
        for preset in presets:
            t_nat_path = os.path.join(bench_root, "Test", "natural", target_name)
            r_nat_path = os.path.join(bench_root, "Test_References", "natural", ref_name)
            r_ret_path = os.path.join(bench_root, "Test_References", "Presets", preset, ref_name)
            gt_path = os.path.join(bench_root, "Test", "Presets", preset, target_name)
            
            if not all(os.path.exists(p) for p in [t_nat_path, r_nat_path, r_ret_path, gt_path]):
                continue
                
            print(f"\n[{target_name} x {preset}]")
            gt = transforms.ToTensor()(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
            
            for strategy in strategies:
                if args.model_type == 'standard':
                    output, t_time = run_tto_standard(
                        model_path=args.meta_weights,
                        target_natural_path_or_tensor=t_nat_path,
                        ref_natural_path_or_tensor=r_nat_path,
                        ref_retouched_path_or_tensor=r_ret_path,
                        device=device,
                        strategy=strategy,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        window_size=args.window_size,
                        hidden_dim=args.hidden_dim
                    )
                else:
                    output, t_time = run_tto_lie(
                        model_path=args.meta_weights,
                        target_natural_path_or_tensor=t_nat_path,
                        ref_natural_path_or_tensor=r_nat_path,
                        ref_retouched_path_or_tensor=r_ret_path,
                        device=device,
                        strategy=strategy,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        window_size=args.window_size,
                        hidden_dim=args.hidden_dim,
                        lambda_ym=args.lambda_ym,
                        use_ssim_lab=True
                    )
                
                if output.shape != gt.shape:
                    output = F.interpolate(output, size=gt.shape[2:], mode='bilinear', align_corners=False)
                    
                psnr = calculate_psnr(output, gt)
                results[strategy].append(psnr)
                print(f"  Strategy: {strategy:<10} | PSNR: {psnr:.4f} dB | Time: {t_time:.2f}s")
                del output
                torch.cuda.empty_cache()
                
    print("\n========================================================")
    print(f"      TTO STRATEGY COMPARISON SUMMARY ({args.steps} Steps)")
    print("========================================================")
    for strategy in strategies:
        avg_psnr = np.mean(results[strategy]) if results[strategy] else 0.0
        print(f"  Average PSNR ({strategy:<10}): {avg_psnr:.4f} dB")
    print("========================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'lie'])
    parser.add_argument('--meta_weights', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--num_presets', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=13)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lambda_ym', type=float, default=0.01)
    args = parser.parse_args()
    main(args)
