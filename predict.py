import argparse
import os

import kornia
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset import CompetitionDataset
from model import InRetouchNR, get_subpixel_sampling_windows


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CompetitionDataset(args.dataset_path)
    print(f"Found {len(dataset)} competition samples.")
    
    os.makedirs(args.output_path, exist_ok=True)

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    for i in tqdm(range(len(dataset))):
        try:
            task = dataset[i]
            sample_id = task['image_name']
            # Using .png for truly lossless saving as requested
            out_file = os.path.join(args.output_path, f"{sample_id}_retouched.png")
            
            if os.path.exists(out_file) and not args.overwrite:
                continue
            
            input_img = Image.open(task['input_path']).convert('RGB')
            ref_in = Image.open(task['ref_input_path']).convert('RGB')
            ref_out = Image.open(task['ref_output_path']).convert('RGB')
            
            input_tensor = to_tensor(input_img).unsqueeze(0).to(device)
            ref_in_tensor = to_tensor(ref_in).unsqueeze(0).to(device)
            ref_out_tensor = to_tensor(ref_out).unsqueeze(0).to(device)
            
            Hr, Wr = ref_in_tensor.shape[2:]
            
            model = InRetouchNR(hidden_dim=128).to(device)
            if os.path.exists(args.load_meta):
                state_dict = torch.load(args.load_meta, map_location=device)
                model.load_state_dict(state_dict)
            else:
                print(f"Warning: Meta weights not found at {args.load_meta}")

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.SmoothL1Loss(beta=0.01)
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-4)
            
            # TTO Loop (500 steps)
            model.train()
            # Receptive field of current model is exactly 29.
            cntx_pad = 14
            cntx_size = args.window_size + 2 * cntx_pad
            ws = args.window_size

            for step in range(args.steps):
                optimizer.zero_grad()
                
                pixel_h, pixel_w = 2.0 / Hr, 2.0 / Wr
                half_cntx_h = (cntx_size / 2.0) * pixel_h
                half_cntx_w = (cntx_size / 2.0) * pixel_w
                
                y_c = torch.rand(args.batch_size, device=device) * (2.0 - 2*half_cntx_h) - (1.0 - half_cntx_h)
                x_c = torch.rand(args.batch_size, device=device) * (2.0 - 2*half_cntx_w) - (1.0 - half_cntx_w)
                centers = torch.stack([x_c, y_c], dim=-1)

                # Subpixel Samples
                patches_in_large_sharp, _, _, _, _ = get_subpixel_sampling_windows(ref_in_tensor, batch_size=args.batch_size, window_size=cntx_size, centers=centers)
                _, patches_in_smooth_13, _, _, _ = get_subpixel_sampling_windows(ref_in_tensor, batch_size=args.batch_size, window_size=ws, centers=centers)
                
                # Target: smooth patches (13x13)
                _, patches_target_smooth, offsets_target, abs_coords_target, _ = get_subpixel_sampling_windows(ref_out_tensor, batch_size=args.batch_size, window_size=ws, centers=centers)
                
                # Use target-aligned offsets and absolute coords for the INR
                model.current_offsets = offsets_target.view(args.batch_size, 2, 1, 1).expand(-1, -1, ws, ws)
                model.current_abs_coords = abs_coords_target
                
                # 3. Forward Pass (FiLM + Residual)
                # content_img is ONLY the 13x13 smooth version
                pred_windows = model(patches_in_smooth_13, x_ctx=patches_in_large_sharp, global_image=ref_in_tensor)
                
                # Loss
                loss_l1 = criterion(pred_windows, patches_target_smooth)
                pred_lab = kornia.color.rgb_to_lab(pred_windows)
                target_lab = kornia.color.rgb_to_lab(patches_target_smooth)
                loss_lab = torch.mean(torch.sqrt(torch.sum((pred_lab - target_lab)**2, dim=1) + 1e-8))
                ssim_val = kornia.metrics.ssim(pred_windows, patches_target_smooth, window_size=5)
                loss_ssim = 1.0 - ssim_val.mean()
                
                # TV Loss
                diff_h = torch.abs(pred_windows[:, :, 1:, :] - pred_windows[:, :, :-1, :])
                diff_w = torch.abs(pred_windows[:, :, :, 1:] - pred_windows[:, :, :, :-1])
                loss_tv = diff_h.mean() + diff_w.mean()
                
                loss = loss_l1 + 0.2 * loss_ssim + 0.05 * loss_lab + 0.001 * loss_tv
                
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Final Prediction
            model.eval()
            model.current_offsets = None 
            model.current_abs_coords = None
            with torch.no_grad():
                final_out = model(input_tensor)
                final_out = final_out.clamp(0, 1)

            # Save Lossless PNG
            to_pil(final_out.squeeze(0)).save(out_file, compress_level=0)

        except Exception as e:
            print(f"Error processing {task['image_name']}: {e}")

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=13)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--load_meta', type=str, default='meta_model_ft.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    main(args)
