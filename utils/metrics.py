"""Centralized evaluation metrics for MetaDC-INR."""

import math
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
import kornia


def calculate_psnr(img1, img2):
    """Calculate PSNR between two torch tensors in [0, 1] range.
    
    Args:
        img1: Tensor of shape (1, C, H, W) or (C, H, W)
        img2: Tensor of shape (1, C, H, W) or (C, H, W)
    
    Returns:
        float: PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-10:
        return 50.0
    return 10 * math.log10(1.0 / mse)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two torch tensors using skimage.
    
    Args:
        img1: Tensor of shape (1, C, H, W)
        img2: Tensor of shape (1, C, H, W)
    
    Returns:
        float: SSIM value
    """
    img1_np = img1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return float(ssim_metric(img1_np, img2_np, data_range=1.0, channel_axis=2))


def calculate_delta_e(pred, target):
    """Mean CIE ΔE (L*a*b*) between predicted and target tensors.
    
    Args:
        pred: Tensor of shape (B, 3, H, W) in [0, 1] RGB
        target: Tensor of shape (B, 3, H, W) in [0, 1] RGB
    
    Returns:
        float: Mean ΔE value
    """
    pred_lab = kornia.color.rgb_to_lab(pred)
    target_lab = kornia.color.rgb_to_lab(target)
    de = torch.sqrt(torch.sum((pred_lab - target_lab) ** 2, dim=1) + 1e-8)
    return de.mean().item()
