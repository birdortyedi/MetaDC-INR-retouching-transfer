"""Centralized loss functions for MetaDC-INR.

Provides:
- CharbonnierLoss: Robust pixel-wise loss matching the paper description
- composite_loss: The full training/TTO objective
- rgb_to_oklab: Differentiable Oklab conversion (kept as utility, not used in main pipeline)
"""

import torch
import torch.nn as nn
import kornia


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (a differentiable variant of L1).
    
    L = mean(sqrt((pred - target)^2 + beta^2))
    
    This is what the paper describes as the primary reconstruction loss.
    Note: This is NOT the same as SmoothL1Loss/Huber loss.
    """
    def __init__(self, beta=0.01):
        super().__init__()
        self.beta_sq = beta ** 2
    
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.beta_sq))


def composite_loss(pred, target, criterion, lambda_ssim=0.2, lambda_lab=0.05, lambda_tv=0.001):
    """Compute the composite TTO/training loss.
    
    L = L_charb + λ_S * L_SSIM + λ_L * L_Lab + λ_T * L_TV
    
    Args:
        pred: Predicted patches (B, 3, H, W)
        target: Target patches (B, 3, H, W)
        criterion: Primary loss function (CharbonnierLoss instance)
        lambda_ssim: Weight for SSIM loss
        lambda_lab: Weight for CIE Lab loss
        lambda_tv: Weight for total variation loss
    
    Returns:
        total_loss: Scalar loss tensor
        loss_dict: Dict of individual loss components for logging
    """
    # Primary reconstruction loss
    loss_recon = criterion(pred, target)
    
    # Structural similarity loss
    loss_ssim = 1.0 - kornia.metrics.ssim(pred, target, window_size=5).mean()
    
    # CIE Lab perceptual color loss
    pred_lab = kornia.color.rgb_to_lab(pred)
    target_lab = kornia.color.rgb_to_lab(target)
    loss_lab = torch.mean(torch.sqrt(torch.sum((pred_lab - target_lab) ** 2, dim=1) + 1e-8))
    
    # Total variation for spatial smoothness
    diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    loss_tv = diff_h.mean() + diff_w.mean()
    
    total = loss_recon + lambda_ssim * loss_ssim + lambda_lab * loss_lab + lambda_tv * loss_tv
    
    loss_dict = {
        'recon': loss_recon.item(),
        'ssim': loss_ssim.item(),
        'lab': loss_lab.item(),
        'tv': loss_tv.item(),
        'total': total.item(),
    }
    
    return total, loss_dict


def rgb_to_oklab(rgb):
    """Differentiable conversion from sRGB [0, 1] to Oklab (L, a, b).
    
    Oklab is perceptually uniform for lightness, chroma, and hue angle.
    Kept as utility for analysis scripts, not used in main training pipeline.
    
    Args:
        rgb: Tensor of shape (B, 3, H, W) in [0, 1]
    
    Returns:
        Tensor of shape (B, 3, H, W) in Oklab space
    """
    m1 = torch.tensor([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ], device=rgb.device, dtype=rgb.dtype)

    m2 = torch.tensor([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ], device=rgb.device, dtype=rgb.dtype)

    rgb_flat = rgb.permute(0, 2, 3, 1)  # [B, H, W, 3]
    # Oklab's M1 is defined on *linear* RGB — decode the sRGB transfer function first.
    lin = torch.where(rgb_flat <= 0.04045,
                      rgb_flat / 12.92,
                      ((rgb_flat + 0.055) / 1.055).clamp(min=1e-8).pow(2.4))
    lms = torch.matmul(lin, m1.T).clamp(min=1e-8)
    lms_cubic = torch.pow(lms, 1.0 / 3.0)
    lab = torch.matmul(lms_cubic, m2.T)
    return lab.permute(0, 3, 1, 2)
