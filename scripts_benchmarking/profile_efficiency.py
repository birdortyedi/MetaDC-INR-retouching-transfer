import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import thop
import argparse
from model import InRetouchNR

def profile_model(model, device):
    print("\n" + "="*50)
    print("      MODEL COMPLEXITY & EFFICIENCY")
    print("="*50)
    
    # 1. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.3f} M")
    
    # 2. MACs per resolution (Analytical scaling)
    # Baseline: 256x256 image
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    macs_256, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
    
    resolutions = {
        "HD (720p)": (720, 1280),
        "Full-HD (1080p)": (1080, 1920),
        "2K (1440p)": (1440, 2560),
        "4K (2160p)": (2160, 3840)
    }
    
    print("\nMACs (Operations) per resolution:")
    print(f"{'Resolution':<20} | {'MACs':<12}")
    print("-" * 35)
    for res_name, (h, w) in resolutions.items():
        # Scale MACs linearly with pixel count
        scale_factor = (h * w) / (256 * 256)
        scaled_macs = macs_256 * scale_factor
        print(f"{res_name:<20} | {scaled_macs / 1e9:>10.2f} G")
    
    # 3. Inference Time across resolutions
    print("\nInference Time (s) per resolution:")
    print(f"{'Resolution':<20} | {'Mean Time':<12} | {'FPS':<8}")
    print("-" * 45)
    
    model.eval()
    with torch.no_grad():
        for res_name, (h, w) in list(resolutions.items())[:2]: # Test HD and FHD if memory allows
            torch.cuda.empty_cache()
            try:
                dummy_in = torch.randn(1, 3, h, w).to(device)
                
                # Warm up
                for _ in range(10):
                    _ = model(dummy_in)
                
                # Measurement
                torch.cuda.synchronize()
                start = time.time()
                iters = 30
                for _ in range(iters):
                    _ = model(dummy_in)
                torch.cuda.synchronize()
                end = time.time()
                
                avg_time = (end - start) / iters # seconds
                fps = 1.0 / avg_time
                print(f"{res_name:<20} | {avg_time:>10.3f} s | {fps:>6.1f}")
                del dummy_in
            except torch.OutOfMemoryError:
                print(f"{res_name:<20} | {'OOM':>10} | {'-':>8}")
            torch.cuda.empty_cache()
    
    print("="*50 + "\n")

def profile_tto_complexity(model, device, batch_size=512, window_size=13):
    print("\n" + "="*50)
    print(f"      TTO STEP COMPLEXITY (Batch: {batch_size})")
    print("="*50)
    
    cntx_size = window_size + 28
    p_smooth = torch.randn(batch_size, 3, window_size, window_size).to(device)
    p_large = torch.randn(batch_size, 3, cntx_size, cntx_size).to(device)
    g_img = torch.randn(1, 3, 512, 512).to(device)
    
    model.current_offsets = torch.zeros(batch_size, 2, window_size, window_size).to(device)
    model.current_abs_coords = torch.zeros(batch_size, 2, window_size, window_size).to(device)
    
    macs, _ = thop.profile(model, inputs=(p_smooth, p_large, g_img), verbose=False)
    print(f"One Step Forward MACs: {macs / 1e9:.3f} G")
    print(f"Total TTO (500 steps forward+backward) ~: {macs * 3 * 500 / 1e12:.3f} TFLOPs")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Profile model efficiency and scalability.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for TTO profiling')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Model hidden dimension')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")

    model = InRetouchNR(hidden_dim=args.hidden_dim).to(device)
    
    profile_model(model, device)
    profile_tto_complexity(model, device, args.batch_size)

if __name__ == "__main__":
    main()
