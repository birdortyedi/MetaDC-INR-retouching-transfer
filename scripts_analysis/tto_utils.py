import os
import time
import torch
import torch.nn.functional as F
import kornia
from torchvision import transforms
from PIL import Image

from model import InRetouchNR as StandardInRetouchNR, get_subpixel_sampling_windows as standard_get_subpixel_sampling_windows
from model_lie import InRetouchNR as LieInRetouchNR, get_subpixel_sampling_windows as lie_get_subpixel_sampling_windows


def run_tto_standard(
    model_path,
    target_natural_path_or_tensor,
    ref_natural_path_or_tensor,
    ref_retouched_path_or_tensor,
    device,
    strategy='selective',
    steps=20,
    lr=1e-3,
    batch_size=512,
    window_size=13,
    hidden_dim=128
):
    """
    Unified Test-Time Optimization (TTO) loop for Standard MetaDC-INR.
    """
    model = StandardInRetouchNR(hidden_dim=hidden_dim).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: weights not found at {model_path}, using random initialization.")

    # Load images if paths are provided
    to_tensor = transforms.ToTensor()
    def _load(inp):
        if isinstance(inp, str):
            return to_tensor(Image.open(inp).convert('RGB')).unsqueeze(0).to(device)
        return inp.to(device)

    t_nat = _load(target_natural_path_or_tensor)
    r_nat = _load(ref_natural_path_or_tensor)
    r_ret = _load(ref_retouched_path_or_tensor)
    
    Hr, Wr = r_nat.shape[2:]
    cntx_size = window_size + 28

    # Apply optimization strategy
    model.train()
    if strategy == 'full':
        tto_params = model.parameters()
    elif strategy == 'selective':
        for param in model.local_context.parameters():
            param.requires_grad = False
        for param in model.global_context.parameters():
            param.requires_grad = False
        tto_params = [p for p in model.parameters() if p.requires_grad]
    elif strategy == 'head_only':
        for name, param in model.named_parameters():
            if "mlp_head" not in name:
                param.requires_grad = False
        tto_params = [p for p in model.parameters() if p.requires_grad]
    else:
        raise ValueError(f"Unknown TTO strategy: {strategy}")

    optimizer = torch.optim.Adam(tto_params, lr=lr)
    criterion = torch.nn.SmoothL1Loss(beta=0.01)
    
    torch.cuda.synchronize()
    tto_start = time.time()
    
    for step in range(steps):
        optimizer.zero_grad()
        
        pixel_h, pixel_w = 2.0 / Hr, 2.0 / Wr
        half_cntx_h = (cntx_size / 2.0) * pixel_h
        half_cntx_w = (cntx_size / 2.0) * pixel_w
        
        y_c = torch.rand(batch_size, device=device) * (2.0 - 2*half_cntx_h) - (1.0 - half_cntx_h)
        x_c = torch.rand(batch_size, device=device) * (2.0 - 2*half_cntx_w) - (1.0 - half_cntx_w)
        centers = torch.stack([x_c, y_c], dim=-1)

        p_large_sharp, _, _, _, _ = standard_get_subpixel_sampling_windows(r_nat, batch_size=batch_size, window_size=cntx_size, centers=centers)
        _, p_smooth_13, _, _, _ = standard_get_subpixel_sampling_windows(r_nat, batch_size=batch_size, window_size=window_size, centers=centers)
        _, p_target_smooth, offsets_target, abs_coords_target, _ = standard_get_subpixel_sampling_windows(r_ret, batch_size=batch_size, window_size=window_size, centers=centers)
        
        model.current_offsets = offsets_target.view(batch_size, 2, 1, 1).expand(-1, -1, window_size, window_size)
        model.current_abs_coords = abs_coords_target
        
        pred = model(p_smooth_13, x_ctx=p_large_sharp, global_image=r_nat)
        
        loss_l1 = criterion(pred, p_target_smooth)
        pred_lab = kornia.color.rgb_to_lab(pred)
        target_lab = kornia.color.rgb_to_lab(p_target_smooth)
        loss_lab = torch.mean(torch.sqrt(torch.sum((pred_lab - target_lab)**2, dim=1) + 1e-8))
        loss_ssim = 1.0 - kornia.metrics.ssim(pred, p_target_smooth, window_size=5).mean()
        
        # TV loss for smoothness
        diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        loss_tv = diff_h.mean() + diff_w.mean()
        
        loss = loss_l1 + 0.2 * loss_ssim + 0.05 * loss_lab + 0.001 * loss_tv
            
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


def run_tto_lie(
    model_path,
    target_natural_path_or_tensor,
    ref_natural_path_or_tensor,
    ref_retouched_path_or_tensor,
    device,
    strategy='selective',
    steps=20,
    lr=1e-3,
    batch_size=512,
    window_size=13,
    hidden_dim=128,
    lambda_ym=0.01,
    use_ssim_lab=False,
    init_params=None,
    init_t=None,
    spline_bins=16
):
    """
    Unified Test-Time Optimization (TTO) loop for Lie-INR.
    """
    model = LieInRetouchNR(hidden_dim=hidden_dim, spline_bins=spline_bins).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: weights not found at {model_path}, using random initialization.")

    # Initialize biases if provided (Log-Map warmstart)
    if init_params is not None and init_t is not None:
        with torch.no_grad():
            model.mlp_head.bias[:9] = init_params
            model.mlp_head.bias[9:12] = init_t

    # Load images if paths are provided
    to_tensor = transforms.ToTensor()
    def _load(inp):
        if isinstance(inp, str):
            return to_tensor(Image.open(inp).convert('RGB')).unsqueeze(0).to(device)
        return inp.to(device)

    t_nat = _load(target_natural_path_or_tensor)
    r_nat = _load(ref_natural_path_or_tensor)
    r_ret = _load(ref_retouched_path_or_tensor)
    
    Hr, Wr = r_nat.shape[2:]
    cntx_size = window_size + 28

    # Apply optimization strategy
    model.train()
    if strategy == 'full':
        tto_params = model.parameters()
    elif strategy == 'selective':
        for param in model.local_context.parameters():
            param.requires_grad = False
        for param in model.global_context.parameters():
            param.requires_grad = False
        tto_params = [p for p in model.parameters() if p.requires_grad]
    elif strategy == 'head_only':
        for name, param in model.named_parameters():
            if "mlp_head" not in name:
                param.requires_grad = False
        tto_params = [p for p in model.parameters() if p.requires_grad]
    else:
        raise ValueError(f"Unknown TTO strategy: {strategy}")

    optimizer = torch.optim.Adam(tto_params, lr=lr)
    criterion = torch.nn.SmoothL1Loss(beta=0.01)
    
    torch.cuda.synchronize()
    tto_start = time.time()
    
    for step in range(steps):
        optimizer.zero_grad()
        
        pixel_h, pixel_w = 2.0 / Hr, 2.0 / Wr
        half_cntx_h = (cntx_size / 2.0) * pixel_h
        half_cntx_w = (cntx_size / 2.0) * pixel_w
        
        y_c = torch.rand(batch_size, device=device) * (2.0 - 2*half_cntx_h) - (1.0 - half_cntx_h)
        x_c = torch.rand(batch_size, device=device) * (2.0 - 2*half_cntx_w) - (1.0 - half_cntx_w)
        centers = torch.stack([x_c, y_c], dim=-1)

        p_large_sharp, _, _, _, _ = lie_get_subpixel_sampling_windows(r_nat, batch_size=batch_size, window_size=cntx_size, centers=centers)
        _, p_smooth_13, _, _, _ = lie_get_subpixel_sampling_windows(r_nat, batch_size=batch_size, window_size=window_size, centers=centers)
        _, p_target_smooth, offsets_target, abs_coords_target, _ = lie_get_subpixel_sampling_windows(r_ret, batch_size=batch_size, window_size=window_size, centers=centers)
        
        model.current_offsets = offsets_target.view(batch_size, 2, 1, 1).expand(-1, -1, window_size, window_size)
        model.current_abs_coords = abs_coords_target
        
        pred = model(p_smooth_13, x_ctx=p_large_sharp, global_image=r_nat)
        
        loss_l1 = criterion(pred, p_target_smooth)
        curv_loss, _ = model.compute_curvature_loss()
        loss = loss_l1 + lambda_ym * curv_loss
        
        if use_ssim_lab:
            pred_lab = kornia.color.rgb_to_lab(pred)
            target_lab = kornia.color.rgb_to_lab(p_target_smooth)
            loss_lab = torch.mean(torch.sqrt(torch.sum((pred_lab - target_lab)**2, dim=1) + 1e-8))
            loss_ssim = 1.0 - kornia.metrics.ssim(pred, p_target_smooth, window_size=5).mean()
            loss += 0.2 * loss_ssim + 0.05 * loss_lab
            
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
