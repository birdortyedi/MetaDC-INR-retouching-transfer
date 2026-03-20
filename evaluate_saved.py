import argparse
import math
import os

import lpips
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms
from tqdm import tqdm


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # LPIPS
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    to_tensor = transforms.ToTensor()
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_files = 0
    
    # Iterate through Results
    presets = [d for d in os.listdir(args.results_path) if os.path.isdir(os.path.join(args.results_path, d))]
    
    for preset in sorted(presets):
        preset_dir = os.path.join(args.results_path, preset)
        images = [f for f in os.listdir(preset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Evaluating {preset} ({len(images)} images)...")
        
        for img_name in tqdm(images):
            res_path = os.path.join(preset_dir, img_name)
            
            # GT path: dataset_path/Benchmark/Test/Presets/preset/img_name
            gt_path = os.path.join(args.dataset_path, 'Benchmark', 'Test', 'Presets', preset, img_name)
            
            if not os.path.exists(gt_path):
                continue
                
            try:
                res_img = Image.open(res_path).convert('RGB')
                gt_img = Image.open(gt_path).convert('RGB')
                
                res_tensor = to_tensor(res_img).unsqueeze(0).to(device)
                gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device)
                
                if res_tensor.shape != gt_tensor.shape:
                    gt_tensor = torch.nn.functional.interpolate(gt_tensor, size=res_tensor.shape[2:], mode='bilinear', align_corners=False)
                
                # PSNR
                psnr = calculate_psnr(res_tensor, gt_tensor)
                total_psnr += psnr
                
                # SSIM
                ssim = calculate_ssim(res_tensor, gt_tensor)
                total_ssim += ssim
                
                # LPIPS
                lpips_val = loss_fn_alex(res_tensor * 2 - 1, gt_tensor * 2 - 1).item()
                total_lpips += lpips_val
                
                num_files += 1
            except Exception as e:
                print(f"Error evaluating {img_name}: {e}")
                
    if num_files > 0:
        print(f"\nFinal Evaluation Results on {num_files} images:")
        print(f"Average PSNR: {total_psnr / num_files:.4f} dB")
        print(f"Average SSIM: {total_ssim / num_files:.4f}")
        print(f"Average LPIPS: {total_lpips / num_files:.4f}")
    else:
        print("No images found for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default='Results')
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
