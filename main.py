import argparse
import math
import os

import kornia
import lpips
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms
from tqdm import tqdm

from dataset import RetouchEvaluationDataset
from model import InRetouchNR, get_subpixel_sampling_windows


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return ssim_metric(img1_np, img2_np, data_range=1.0, channel_axis=2)


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RetouchEvaluationDataset(args.dataset_path)
    print(f"Found {len(dataset)} tasks.")
    
    os.makedirs(args.output_path, exist_ok=True)

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_evaluated = 0
    
    for i in tqdm(range(len(dataset))):
        try:
            task = dataset[i]
            preset = task['preset']
            img_name = task['image_name']
            out_file = os.path.join(args.output_path, preset, img_name)
            
            if os.path.exists(out_file) and not args.overwrite:
                continue
                
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            
            input_img = Image.open(task['input_path']).convert('RGB')
            ref_in = Image.open(task['ref_input_path']).convert('RGB')
            ref_out = Image.open(task['ref_output_path']).convert('RGB')
            
            input_tensor = to_tensor(input_img).unsqueeze(0).to(device)
            ref_in_tensor = to_tensor(ref_in).unsqueeze(0).to(device)
            ref_out_tensor = to_tensor(ref_out).unsqueeze(0).to(device)
            
            Hr, Wr = ref_in_tensor.shape[2:]
            
            model = InRetouchNR(hidden_dim=128).to(device)
            
            if args.load_meta and os.path.exists(args.load_meta):
                try:
                    state_dict = torch.load(args.load_meta, map_location=device)
                    model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Meta load error: {e}")

            model.to(device)
        
            # Freeze CNN feature extractors during long TTO to prevent artifacts
            for param in model.local_context.parameters():
                param.requires_grad = False
            for param in model.global_context.parameters():
                param.requires_grad = False
                
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            criterion = nn.SmoothL1Loss(beta=0.01)
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-4)
            
            cntx_pad = 14
            cntx_size = args.window_size + 2 * cntx_pad
            ws = args.window_size
            
            for step in range(args.steps):
                optimizer.zero_grad()
                
                pixel_h, pixel_w = 2.0 / Hr, 2.0 / Wr
                half_cntx_h, half_cntx_w = (cntx_size / 2.0) * pixel_h, (cntx_size / 2.0) * pixel_w
                
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
                
                pred_windows = model(patches_in_smooth_13, x_ctx=patches_in_large_sharp, global_image=ref_in_tensor)
                loss_l1 = criterion(pred_windows, patches_target_smooth)
                pred_lab = kornia.color.rgb_to_lab(pred_windows)
                target_lab = kornia.color.rgb_to_lab(patches_target_smooth)
                loss_lab = torch.mean(torch.sqrt(torch.sum((pred_lab - target_lab)**2, dim=1) + 1e-8))
                ssim_val = kornia.metrics.ssim(pred_windows, patches_target_smooth, window_size=5)
                loss_ssim = 1.0 - ssim_val.mean()
                
                # TV Loss (Kill the remaining grid noise)
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
            with torch.no_grad():
                final_out = model(input_tensor)
                final_out = final_out.clamp(0, 1)
                
            gt_img = Image.open(task['gt_path']).convert('RGB')
            gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device)
            
            psnr = calculate_psnr(final_out, gt_tensor)
            ssim_val = calculate_ssim(final_out, gt_tensor)
            lpips_val = loss_fn_alex(final_out, gt_tensor).item()
            
            total_psnr += psnr
            total_ssim += ssim_val
            total_lpips += lpips_val
            num_evaluated += 1
            
            if not args.skip_save:
                to_pil(final_out.squeeze(0)).save(out_file)

        except Exception as e:
            print(f"Error processing {task['image_name']}: {e}")

    if num_evaluated > 0:
        print(f"\nFinal Evaluation Results on {num_evaluated} images:")
        print(f"Average PSNR: {total_psnr/num_evaluated:.4f} dB")
        print(f"Average SSIM: {total_ssim/num_evaluated:.4f}")
        print(f"Average LPIPS: {total_lpips/num_evaluated:.4f}")

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=484)
    parser.add_argument('--window_size', type=int, default=13)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--load_meta', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip_save', action='store_true')
    args = parser.parse_args()
    main(args)
