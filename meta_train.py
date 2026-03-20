import argparse
import copy
import os

import kornia
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset import RetouchDataset
from model import InRetouchNR, get_subpixel_sampling_windows


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RetouchDataset(args.dataset_path)
    print(f"Dataset initialized with {len(dataset)} total tasks.")
    
    meta_model = InRetouchNR(hidden_dim=128).to(device)
    
    to_tensor = transforms.ToTensor()
    l1_criterion = nn.SmoothL1Loss(beta=0.01)
    
    # Meta-training hyperparameters
    meta_lr = args.meta_lr  # epsilon in Reptile
    
    # Create visualization directory
    vis_dir = "meta_vis"
    os.makedirs(vis_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nMeta-Epoch {epoch+1}/{args.epochs}")
        
        # Shuffle tasks
        indices = torch.randperm(len(dataset))
        
        pbar = tqdm(range(len(dataset)))
        for i in pbar:
            task = dataset[indices[i]]
            
            input_img = Image.open(task['input_path']).convert('RGB')
            target_img = Image.open(task['target_path']).convert('RGB')
            
            input_tensor = to_tensor(input_img).unsqueeze(0).to(device)
            target_tensor = to_tensor(target_img).unsqueeze(0).to(device)

            # Data Augmentation
            if torch.rand(1) > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[3])
                target_tensor = torch.flip(target_tensor, dims=[3])
            if torch.rand(1) > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[2])
                target_tensor = torch.flip(target_tensor, dims=[2])
            
            # Image dimensions for sub-pixel sampling normalization
            H, W = input_tensor.shape[2:]
            
            # --- Check if visualization is needed for this task ---
            do_vis = (i == 0 or (i + 1) % args.vis_freq == 0)
            
            # Trajectory capture for optimization analysis (Discovery Swish: 3, 6, 12)
            vis_trajectory = []
            vis_steps_list = [0, 3, 6, args.inner_steps]
            
            # INNER LOOP (Task Adaptation)
            task_model = copy.deepcopy(meta_model)
            task_optimizer = optim.Adam(task_model.parameters(), lr=args.inner_lr)
            
            ws = args.window_size
            cntx_pad = 14
            cntx_size = ws + 2 * cntx_pad
            pixel_h, pixel_w = 2.0 / H, 2.0 / W
            half_cntx_h, half_cntx_w = (cntx_size / 2.0) * pixel_h, (cntx_size / 2.0) * pixel_w
            
            for k in range(args.inner_steps + 1):  # Ran one extra to capture step 'inner_steps'
                # --- Visualization Save ---
                if do_vis and k in vis_steps_list:
                    task_model.eval()
                    task_model.current_offsets = None  # Set offsets to None for full image inference
                    task_model.current_abs_coords = None
                    with torch.no_grad():
                        traj_pred = task_model(input_tensor)
                        vis_trajectory.append(traj_pred)
                
                if k == args.inner_steps:
                    break
                    
                task_model.train()
                task_optimizer.zero_grad()
                
                # 1. Sample Centers for Sub-pixel Training
                pixel_h, pixel_w = 2.0 / H, 2.0 / W
                half_cntx_h = (cntx_size / 2.0) * pixel_h
                half_cntx_w = (cntx_size / 2.0) * pixel_w
                
                y_c = torch.rand(args.batch_size, device=device) * (2.0 - 2*half_cntx_h) - (1.0 - half_cntx_h)
                x_c = torch.rand(args.batch_size, device=device) * (2.0 - 2*half_cntx_w) - (1.0 - half_cntx_w)
                centers = torch.stack([x_c, y_c], dim=-1)

                # Subpixel Samples
                patches_in_large_sharp, _, _, _, _ = get_subpixel_sampling_windows(input_tensor, batch_size=args.batch_size, window_size=cntx_size, centers=centers)
                _, patches_in_smooth_13, _, _, _ = get_subpixel_sampling_windows(input_tensor, batch_size=args.batch_size, window_size=ws, centers=centers)
                
                # Target: smooth patches (13x13)
                _, patches_target_smooth, offsets_target, abs_coords_target, _ = get_subpixel_sampling_windows(target_tensor, batch_size=args.batch_size, window_size=ws, centers=centers)
                
                # Use target-aligned offsets and absolute coords for the INR
                task_model.current_offsets = offsets_target.view(args.batch_size, 2, 1, 1).expand(-1, -1, ws, ws)
                task_model.current_abs_coords = abs_coords_target
                
                pred_windows = task_model(patches_in_smooth_13, x_ctx=patches_in_large_sharp, global_image=input_tensor)
                
                loss_l1 = l1_criterion(pred_windows, patches_target_smooth)
                pred_lab = kornia.color.rgb_to_lab(pred_windows)
                target_lab = kornia.color.rgb_to_lab(patches_target_smooth)
                loss_lab = torch.mean(torch.sqrt(torch.sum((pred_lab - target_lab)**2, dim=1) + 1e-8))
                ssim_val = kornia.metrics.ssim(pred_windows, patches_target_smooth, window_size=5)
                loss_ssim = 1.0 - ssim_val.mean()
                
                # TV Loss (Smoothness prior)
                diff_h = torch.abs(pred_windows[:, :, 1:, :] - pred_windows[:, :, :-1, :])
                diff_w = torch.abs(pred_windows[:, :, :, 1:] - pred_windows[:, :, :, :-1])
                loss_tv = diff_h.mean() + diff_w.mean()
                
                loss = loss_l1 + 0.2 * loss_ssim + 0.05 * loss_lab + 0.001 * loss_tv
                
                loss.backward()
                task_optimizer.step()

            # --- Visualization Save ---
            if do_vis:
                # Create aTrajectory strip: Input | GT | Step 0 | Step 5 | Step 10 | Final
                strip_list = [input_tensor, target_tensor] + vis_trajectory
                vis_img = torch.cat(strip_list, dim=3)
                to_pil = transforms.ToPILImage()
                to_pil(vis_img.squeeze(0).cpu().clamp(0, 1)).save(os.path.join(vis_dir, f"step_{i+1:05d}.png"))

            # Meta-Update
            # Phi = Phi + epsilon * (Phi' - Phi)
            current_meta_lr = args.meta_lr * (1.0 - (epoch * len(dataset) + i) / (args.epochs * len(dataset)))
            current_meta_lr = max(current_meta_lr, 0.001)

            with torch.no_grad():
                for meta_param, task_param in zip(meta_model.parameters(), task_model.parameters()):
                    meta_param.data.add_(task_param.data - meta_param.data, alpha=current_meta_lr)
            
            pbar.set_description(f"Loss: {loss.item():.4f} | mLR: {current_meta_lr:.4f}")
            
            del task_model
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
            
            # Save periodic checkpoints
            if (i + 1) % args.save_freq == 0:
                torch.save(meta_model.state_dict(), f"meta_model_ft_latest.pth")
                
    torch.save(meta_model.state_dict(), "meta_model_ft.pth")
    print("Meta-training complete. Saved to meta_model_ft.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--inner_steps', type=int, default=12)
    parser.add_argument('--inner_lr', type=float, default=1e-3)
    parser.add_argument('--meta_lr', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=13)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--vis_freq', type=int, default=4000)
    args = parser.parse_args()
    main(args)
