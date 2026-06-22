import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import time
import torch
import thop
import pandas as pd

sys.path.insert(0, '/home/birdortyedi/inr-retouching')
from model import InRetouchNR, get_subpixel_sampling_windows
import torch.nn as nn
import torch.optim as optim

def benchmark_inference(model_name, model, input_tensors, device, iters=30):
    model.eval()
    model.to(device)
    input_tensors = [t.to(device) for t in input_tensors]
    
    macs, params = thop.profile(model, inputs=tuple(input_tensors), verbose=False)
    
    with torch.no_grad():
        for _ in range(5):
            _ = model(*input_tensors)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = model(*input_tensors)
        torch.cuda.synchronize()
        end = time.time()
        
    avg_time_ms = ((end - start) / iters) * 1000
    fps = 1000.0 / avg_time_ms
    
    return params / 1e6, macs / 1e9, avg_time_ms, fps

def benchmark_tto(model, device, steps=10, batch_size=512, window_size=13):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss(beta=0.01)
    
    Hr, Wr = 720, 1280
    cntx_size = window_size + 28
    
    # Dummy data
    ref_in = torch.randn(1, 3, Hr, Wr, device=device)
    ref_out = torch.randn(1, 3, Hr, Wr, device=device)
    
    pixel_h, pixel_w = 2.0 / Hr, 2.0 / Wr
    half_cntx_h = (cntx_size / 2.0) * pixel_h
    half_cntx_w = (cntx_size / 2.0) * pixel_w
    
    # Pre-generate centers for timing
    y_c = torch.rand(batch_size, device=device) * (2.0 - 2*half_cntx_h) - (1.0 - half_cntx_h)
    x_c = torch.rand(batch_size, device=device) * (2.0 - 2*half_cntx_w) - (1.0 - half_cntx_w)
    centers = torch.stack([x_c, y_c], dim=-1)
    
    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        patches_in_large, _, _, _, _ = get_subpixel_sampling_windows(ref_in, batch_size, cntx_size, centers)
        _, patches_in_smooth, _, _, _ = get_subpixel_sampling_windows(ref_in, batch_size, window_size, centers)
        _, patches_target, offsets, abs_coords, _ = get_subpixel_sampling_windows(ref_out, batch_size, window_size, centers)
        
        model.current_offsets = offsets.view(batch_size, 2, 1, 1).expand(-1, -1, window_size, window_size)
        model.current_abs_coords = abs_coords
        pred = model(patches_in_smooth, x_ctx=patches_in_large, global_image=ref_in)
        loss = criterion(pred, patches_target)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        patches_in_large, _, _, _, _ = get_subpixel_sampling_windows(ref_in, batch_size, cntx_size, centers)
        _, patches_in_smooth, _, _, _ = get_subpixel_sampling_windows(ref_in, batch_size, window_size, centers)
        _, patches_target, offsets, abs_coords, _ = get_subpixel_sampling_windows(ref_out, batch_size, window_size, centers)
        
        model.current_offsets = offsets.view(batch_size, 2, 1, 1).expand(-1, -1, window_size, window_size)
        model.current_abs_coords = abs_coords
        pred = model(patches_in_smooth, x_ctx=patches_in_large, global_image=ref_in)
        loss = criterion(pred, patches_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / steps

def benchmark_tto_competitor(model, device, steps=10, batch_size=484, window_size=12):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.L1Loss()
    
    # Dummy data representing the sampled (p, s) patches
    ref_in = torch.randn(batch_size, 5, window_size, window_size, device=device)
    ref_out = torch.randn(batch_size, 3, window_size, window_size, device=device)
    
    for _ in range(2):
        optimizer.zero_grad()
        pred = model(ref_in)
        loss = criterion(pred, ref_out)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(ref_in)
        loss = criterion(pred, ref_out)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / steps

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}...")
    
    results = []
    
    # 1. INReTouch (1st Place)
    try:
        sys.path.insert(0, '/home/birdortyedi/PycharmProjects/InRetouch/basicsr/models/archs')
        import CNNDWSIREN_split_arch as arch1
        inretouch = arch1.CNNDWSplitSiren(n_input_p=2, n_input_s=3, n_output_dims=3, sin_w=1, n_neurons=64, 
                     n_hidden_p=1, n_hidden_s=1, n_hidden_m=1, use_skip=True)
        p, m, t, f = benchmark_inference("INReTouch", inretouch, [torch.randn(1, 5, 720, 1280)], device)
        tto_inretouch = benchmark_tto_competitor(inretouch, device, steps=20) * 1000
        results.append({"Model": "INReTouch (1st)", "Params (M)": p, "MACs (G)": m, "Inference (ms)": t, "TTO Steps": 1000, "TTO Total (s)": tto_inretouch})
    except (ImportError, ModuleNotFoundError):
        print("Skipping INReTouch profiling (archive/dependency not found).")
    finally:
        if '/home/birdortyedi/PycharmProjects/InRetouch/basicsr/models/archs' in sys.path:
            sys.path.remove('/home/birdortyedi/PycharmProjects/InRetouch/basicsr/models/archs')
        
    # 2. Team A/E (2nd Place)
    try:
        sys.path.insert(0, '/home/birdortyedi/Downloads/InRetouch_backup/InRetouch/basicsr/models/archs')
        import CNNDWSIREN_split_arch as arch2
        teama = arch2.CNNDWSplitSiren(n_input_p=2, n_input_s=3, n_output_dims=3, sin_w=1, n_neurons=64, 
                     n_hidden_p=1, n_hidden_s=1, n_hidden_m=1, use_skip=True)
        p, m, t, f = benchmark_inference("Team A", teama, [torch.randn(1, 5, 720, 1280)], device)
        tto_teama = benchmark_tto_competitor(teama, device, steps=20) * 1500
        results.append({"Model": "Team A (2nd)", "Params (M)": p, "MACs (G)": m, "Inference (ms)": t, "TTO Steps": 1500, "TTO Total (s)": tto_teama})
    except (ImportError, ModuleNotFoundError):
        print("Skipping Team A/E profiling (archive/dependency not found).")
    finally:
        if '/home/birdortyedi/Downloads/InRetouch_backup/InRetouch/basicsr/models/archs' in sys.path:
            sys.path.remove('/home/birdortyedi/Downloads/InRetouch_backup/InRetouch/basicsr/models/archs')
    
    # 3. Ours (h=128)
    ours_128 = InRetouchNR(hidden_dim=128).to(device)
    p, m, t, f = benchmark_inference("Ours", ours_128, [torch.randn(1, 3, 720, 1280)], device)
    tto_time = benchmark_tto(ours_128, device, steps=20) * 100
    results.append({"Model": "MetaDC-INR (h128)", "Params (M)": p, "MACs (G)": m, "Inference (ms)": t, "TTO Steps": 100, "TTO Total (s)": tto_time})
    
    # 4. Ours (h=64)
    ours_64 = InRetouchNR(hidden_dim=64).to(device)
    p, m, t, f = benchmark_inference("Ours", ours_64, [torch.randn(1, 3, 720, 1280)], device)
    tto_time_64 = benchmark_tto(ours_64, device, steps=20) * 100
    results.append({"Model": "MetaDC-INR (h64)", "Params (M)": p, "MACs (G)": m, "Inference (ms)": t, "TTO Steps": 100, "TTO Total (s)": tto_time_64})
    
    # 5. Ours Cross-Attention (h=128)
    try:
        from model_cross_attention import InRetouchNR as InRetouchCrossAttn
        ours_128_attn = InRetouchCrossAttn(hidden_dim=128).to(device)
        p, m, t, f = benchmark_inference("Ours Attn", ours_128_attn, [torch.randn(1, 3, 720, 1280)], device)
        tto_time_attn = benchmark_tto(ours_128_attn, device, steps=20) * 100
        results.append({"Model": "MetaDC-INR Attn (h128)", "Params (M)": p, "MACs (G)": m, "Inference (ms)": t, "TTO Steps": 100, "TTO Total (s)": tto_time_attn})
    except (ImportError, ModuleNotFoundError):
        print("Skipping Cross-Attention profiling (model_cross_attention.py not found).")
        
    # 6. Ours Bilateral Attention (h=128)
    try:
        from model_bilateral_attention import InRetouchNR as InRetouchBilateral
        ours_128_bilateral = InRetouchBilateral(hidden_dim=128).to(device)
        p, m, t, f = benchmark_inference("Ours Bilateral", ours_128_bilateral, [torch.randn(1, 3, 720, 1280)], device)
        tto_time_bilateral = benchmark_tto(ours_128_bilateral, device, steps=20) * 100
        results.append({"Model": "MetaDC-INR Bilateral (h128)", "Params (M)": p, "MACs (G)": m, "Inference (ms)": t, "TTO Steps": 100, "TTO Total (s)": tto_time_bilateral})
    except (ImportError, ModuleNotFoundError):
        print("Skipping Bilateral Attention profiling (model_bilateral_attention.py not found).")
    
    df = pd.DataFrame(results)
    df = df.round(2)
    print("\n" + "="*90)
    print("      TIME & COMPLEXITY BENCHMARK (Resolution: 1280x720 HD)")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
