#!/usr/bin/env python
"""Supervised pre-training, timed: training-cost comparison with Reptile meta-training (Section 4.4.3).

Setup: the meta-training pairs without Benchmark/Test, 12 epochs, one update per pair per
epoch. One update = one gradient step on 512 sub-pixel patches from a single image pair, built exactly as a
Reptile inner step in meta_train.py: same sampler, same augmentation, same loss and loss weights, lr 1e-3.

Imports the repository primitives unchanged. Writes only into the current working directory; never touches weights/.

    python scripts_benchmarking/supervised_timing.py --dataset_path <RTD> --optimizer adam --epochs 12

Logs per epoch: wall time, updates, sampled per-update data/compute split, torch peak memory.
"""
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import kornia
import torch
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from dataset import RetouchDataset
from model import InRetouchNR, get_subpixel_sampling_windows
from utils.losses import CharbonnierLoss


def make_optimizer(name, params, lr):
    if name == "sgd":
        return optim.SGD(params, lr=lr)
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "adamw":
        return optim.AdamW(params, lr=lr)
    raise ValueError(name)


def main(args):
    device = torch.device(f"cuda:{args.gpu}")
    items = [t for t in RetouchDataset(args.dataset_path).items
             if f"{os.sep}Benchmark{os.sep}" not in t["input_path"]]
    print(f"training pairs (Benchmark/Test excluded): {len(items)}")

    model = InRetouchNR(hidden_dim=128).to(device)
    opt = make_optimizer(args.optimizer, model.parameters(), args.lr)
    start_epoch = 0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_epoch = ck["epoch"]
        print(f"resumed from {args.resume_from} at epoch {start_epoch}")

    to_tensor = transforms.ToTensor()
    l1 = CharbonnierLoss(beta=0.01)
    ws, B = args.window_size, args.batch_size
    cntx_size = ws + 2 * 14

    log_path = f"{args.out_prefix}_timing.csv"
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline=""); log = csv.writer(logf)
    if new_log:
        log.writerow(["epoch", "updates", "wall_s", "data_s_per_update", "compute_s_per_update", "probes", "peak_alloc_mib", "peak_reserved_mib"])

    done = 0
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.reset_peak_memory_stats(device)
        g = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(items), generator=g)
        data_s = comp_s = 0.0; n_probe = 0; ep_updates = 0
        t_epoch = time.time()
        model.train()
        for n, idx in enumerate(order.tolist()):
            if args.max_updates and done >= args.max_updates:
                break
            t0 = time.time()
            task = items[idx]
            x = to_tensor(Image.open(task["input_path"]).convert("RGB")).unsqueeze(0).to(device)
            y = to_tensor(Image.open(task["target_path"]).convert("RGB")).unsqueeze(0).to(device)
            if torch.rand(1) > 0.5:
                x, y = torch.flip(x, dims=[3]), torch.flip(y, dims=[3])
            if torch.rand(1) > 0.5:
                x, y = torch.flip(x, dims=[2]), torch.flip(y, dims=[2])
            probe = (n % args.probe_every == 0)
            if probe:
                torch.cuda.synchronize(device); t1 = time.time()

            H, W = x.shape[2:]
            hh, hw = (cntx_size / 2.0) * (2.0 / H), (cntx_size / 2.0) * (2.0 / W)
            yc = torch.rand(B, device=device) * (2.0 - 2 * hh) - (1.0 - hh)
            xc = torch.rand(B, device=device) * (2.0 - 2 * hw) - (1.0 - hw)
            centers = torch.stack([xc, yc], dim=-1)
            p_large, _, _, _, _ = get_subpixel_sampling_windows(x, batch_size=B, window_size=cntx_size, centers=centers)
            _, p_in, _, _, _ = get_subpixel_sampling_windows(x, batch_size=B, window_size=ws, centers=centers)
            _, p_tg, off, absc, _ = get_subpixel_sampling_windows(y, batch_size=B, window_size=ws, centers=centers)
            model.current_offsets = off.view(B, 2, 1, 1).expand(-1, -1, ws, ws)
            model.current_abs_coords = absc

            opt.zero_grad()
            pred = model(p_in, x_ctx=p_large, global_image=x)
            loss_l1 = l1(pred, p_tg)
            pl, tl = kornia.color.rgb_to_lab(pred), kornia.color.rgb_to_lab(p_tg)
            loss_lab = torch.mean(torch.sqrt(torch.sum((pl - tl) ** 2, dim=1) + 1e-8))
            loss_ssim = 1.0 - kornia.metrics.ssim(pred, p_tg, window_size=5).mean()
            loss_tv = (torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
                       + torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean())
            loss = loss_l1 + 0.2 * loss_ssim + 0.05 * loss_lab + 0.001 * loss_tv
            loss.backward(); opt.step()
            if probe:
                torch.cuda.synchronize(device); t2 = time.time()
                data_s += t1 - t0; comp_s += t2 - t1; n_probe += 1
            done += 1; ep_updates += 1
            if (n + 1) % 1000 == 0:
                print(f"epoch {epoch + 1} {n + 1}/{len(items)} loss {loss.item():.4f} "
                      f"{(time.time() - t_epoch) / (n + 1):.3f} s/update", flush=True)
        wall = time.time() - t_epoch
        log.writerow([epoch + 1, ep_updates, f"{wall:.1f}", f"{data_s / max(n_probe, 1):.4f}",
                      f"{comp_s / max(n_probe, 1):.4f}", n_probe,
                      f"{torch.cuda.max_memory_allocated(device) / 2**20:.0f}",
                      f"{torch.cuda.max_memory_reserved(device) / 2**20:.0f}"]); logf.flush()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1},
                   f"{args.out_prefix}_resume.pth")
        print(f"epoch {epoch + 1} done: {wall / 3600:.2f} h", flush=True)
        if args.max_updates and done >= args.max_updates:
            break
    logf.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adam")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--window_size", type=int, default=13)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--probe_every", type=int, default=50)
    p.add_argument("--max_updates", type=int, default=0, help="stop early (smoke test); 0 = full run")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--out_prefix", default="sup_timing")
    main(p.parse_args())
