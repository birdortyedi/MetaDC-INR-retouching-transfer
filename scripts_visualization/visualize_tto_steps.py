import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import kornia

from dataset import CompetitionDataset
from model import InRetouchNR, get_subpixel_sampling_windows

import lpips

def to_np(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

def float_to_uint8(img_np):
    return (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

def compute_metrics(pred_uint8, gt_uint8, lpips_model, device):
    # Convert to tensor [-1, 1] for LPIPS
    pred_t = torch.from_numpy(pred_uint8).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device) * 2 - 1
    gt_t = torch.from_numpy(gt_uint8).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device) * 2 - 1
    
    with torch.no_grad():
        lpips_val = lpips_model(pred_t, gt_t).item()
        # DeltaE
        pred_lab = kornia.color.rgb_to_lab(torch.from_numpy(pred_uint8).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device))
        gt_lab = kornia.color.rgb_to_lab(torch.from_numpy(gt_uint8).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device))
        delta_e = torch.sqrt(torch.sum((pred_lab - gt_lab)**2, dim=1)).mean().item()
    return lpips_val, delta_e

def run_tto_visualize(ref_in_tensor, ref_out_tensor, model, device, args, target_steps):
    Hr, Wr = ref_in_tensor.shape[2:]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss(beta=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-4)
    
    cntx_pad = 14
    cntx_size = args.window_size + 2 * cntx_pad
    ws = args.window_size
    
    results = {}
    model.train()
    for step in range(1, args.steps + 1):
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
        loss_tv = torch.abs(pred_windows[:, :, 1:, :] - pred_windows[:, :, :-1, :]).mean() + \
                  torch.abs(pred_windows[:, :, :, 1:] - pred_windows[:, :, :, :-1]).mean()
        loss = loss_l1 + 0.2 * loss_ssim + 0.05 * loss_lab + 0.001 * loss_tv
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step in target_steps:
            model.eval()
            with torch.no_grad():
                model.current_offsets = None
                model.current_abs_coords = None
                out = model(ref_in_tensor, x_ctx=ref_in_tensor, global_image=ref_in_tensor)
                results[step] = float_to_uint8(to_np(out))
            model.train()
    return results

def draw_overlay(img_uint8, label_lines, font, font_size):
    pil_img = Image.fromarray(img_uint8).convert('RGBA')
    overlay = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Text padding and positioning
    padding = 10
    line_height = font_size + 5
    box_h = len(label_lines) * line_height + padding * 2
    
    # Max width for box
    max_w = 0
    for line in label_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_w = max(max_w, bbox[2] - bbox[0])
    box_w = max_w + padding * 2
    
    # Position: bottom left
    bx0, by0 = 10, pil_img.height - box_h - 10
    bx1, by1 = bx0 + box_w, by0 + box_h
    
    # Draw semi-transparent background box
    draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0, 160))
    
    # Draw text
    for i, line in enumerate(label_lines):
        draw.text((bx0 + padding, by0 + padding + i * line_height), 
                  line, fill=(255, 255, 255, 255), font=font)
    
    return np.array(Image.alpha_composite(pil_img, overlay).convert('RGB'))

def stitch_row(images, labels, font, font_size, gap=2):
    # Ensure all images have the same height
    h = images[0].shape[0]
    processed_panels = []
    for i, img in enumerate(images):
        if img.shape[0] != h:
            pil_p = Image.fromarray(img)
            new_w = int(round(img.shape[1] * h / img.shape[0]))
            img = np.array(pil_p.resize((new_w, h), Image.LANCZOS))
            
        # Draw overlay on all except Input/GT if needed, or all
        label_lines = labels[i].split('\n')
        processed_panels.append(draw_overlay(img, label_lines, font, font_size))
    
    total_w = sum(p.shape[1] for p in processed_panels) + gap * (len(processed_panels) - 1)
    canvas = np.ones((h, total_w, 3), dtype=np.uint8) * 255
    
    curr_x = 0
    for p in processed_panels:
        canvas[:, curr_x:curr_x+p.shape[1]] = p
        curr_x += p.shape[1] + gap
        
    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--meta_weights', type=str, default='weights/meta_model_ft.pth')
    parser.add_argument('--output_dir', type=str, default='tto_visuals')
    parser.add_argument('--samples', type=str, default='', help='Comma-separated sample names')
    parser.add_argument('--num_samples', type=int, default=25, help='Number of random samples to pick if --samples is empty')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=13)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = CompetitionDataset(args.dataset_path)
    
    sample_ids = []
    if args.samples:
        sample_ids = [s.strip() for s in args.samples.split(',')]
    
    if len(sample_ids) < args.num_samples:
        import random
        random.seed(args.seed)
        all_names = [dataset[i]['image_name'] for i in range(len(dataset))]
        # Exclude already picked ones
        remaining = [n for n in all_names if n not in sample_ids]
        random.shuffle(remaining)
        needed = args.num_samples - len(sample_ids)
        sample_ids.extend(remaining[:needed])
    
    print(f"Selected {len(sample_ids)} samples: {', '.join(sample_ids)}")
    
    target_steps = [1, 5, 10, 50, 100, 500]
    font_size = 36
    font = None
    font_paths = [
        '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, font_size)
            break
    if font is None: font = ImageFont.load_default()
    
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    to_tensor = transforms.ToTensor()
    
    for sid in sample_ids:
        task = next((d for d in dataset if d['image_name'] == sid), None)
        if task is None: continue
        
        print(f"Processing {sid}...")
        ref_in = Image.open(task['ref_input_path']).convert('RGB')
        ref_out = Image.open(task['ref_output_path']).convert('RGB')
        
        ref_in_tensor = to_tensor(ref_in).unsqueeze(0).to(device)
        ref_out_tensor = to_tensor(ref_out).unsqueeze(0).to(device)
        
        model = InRetouchNR(hidden_dim=128).to(device)
        model.load_state_dict(torch.load(args.meta_weights, map_location=device))
        
        torch.manual_seed(args.seed)
        step_results = run_tto_visualize(ref_in_tensor, ref_out_tensor, model, device, args, target_steps)
        
        input_uint8 = float_to_uint8(to_np(ref_in_tensor))
        gt_uint8 = float_to_uint8(to_np(ref_out_tensor))
        h_gt, w_gt = gt_uint8.shape[:2]
        
        images = [input_uint8]
        labels = ["Input"]
        
        for step in target_steps:
            res_uint8 = step_results[step]
            # Ensure size match for metrics
            if res_uint8.shape[:2] != (h_gt, w_gt):
                pil_res = Image.fromarray(res_uint8)
                res_uint8 = np.array(pil_res.resize((w_gt, h_gt), Image.LANCZOS))
            
            lpips_val, delta_e = compute_metrics(res_uint8, gt_uint8, lpips_model, device)
            images.append(res_uint8)
            labels.append(f"Step {step}\nLPIPS: {lpips_val:.4f}\nΔE: {delta_e:.2f}")
            
        # Add GT at the end
        images.append(gt_uint8)
        labels.append("GT")
            
        row_img = stitch_row(images, labels, font, font_size, gap=3)
        out_path = os.path.join(args.output_dir, f"{sid}_tto_progress.png")
        Image.fromarray(row_img).save(out_path)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
