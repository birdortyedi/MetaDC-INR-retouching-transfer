#!/usr/bin/env python
"""MetaDC-INR zero-shot on IFFI (Table 6), under the Table 1 protocol.

Test-time optimization is benchmark.py's own run_tto, imported and called unchanged (all parameters
adapted, Adam at lr 1e-3, batch 512, window 13), so the IFFI numbers are directly comparable with
Table 1. The only intervention is that, inside benchmark.py, `Image.open` returns the 768 px-capped
image (iffi_tasks.capped_pil). Metrics use benchmark.py's functions: PSNR, SSIM, LPIPS-Alex on [-1, 1].

    python scripts_benchmarking/benchmark_iffi.py --iffi_root <IFFI>/test --steps_list 10,100,200,500 \
        --out_dir results/iffi

Writes a per-task CSV per budget and a summary JSON to --out_dir.
"""
import argparse, csv, importlib.util, json, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F  # noqa: E402
from torchvision import transforms  # noqa: E402
import iffi_tasks  # noqa: E402

spec = importlib.util.spec_from_file_location("benchmark", os.path.join(REPO, "scripts_benchmarking", "benchmark.py"))
benchmark = importlib.util.module_from_spec(spec); spec.loader.exec_module(benchmark)


class _CappedImage:  # stands in for PIL.Image inside benchmark.py only
    @staticmethod
    def open(path):
        return iffi_tasks.capped_pil(path)


benchmark.Image = _CappedImage


def main(a):
    dev = torch.device(f"cuda:{a.gpu}")
    assert os.path.exists(a.meta_weights), a.meta_weights   # run_tto would otherwise fall back to random init
    lp = benchmark.lpips.LPIPS(net="alex").to(dev)
    tt = transforms.ToTensor()
    T = iffi_tasks.tasks(a.iffi_root)
    if a.limit > 0:
        T = T[:a.limit]  # smoke tests only
    print(f"IFFI tasks: {len(T)}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {}
    for steps in [int(s) for s in a.steps_list.split(",")]:
        path = os.path.join(a.out_dir, f"metadc_iffi_{steps}steps.csv")
        rows, t0 = [], time.time()
        for n, t in enumerate(T):
            out, secs = benchmark.run_tto(a.meta_weights, t["t_nat"], t["r_nat"], t["r_ret"], dev, steps=steps,
                                          batch_size=a.batch_size, window_size=a.window_size, hidden_dim=128)
            gt = tt(iffi_tasks.capped_pil(t["gt"])).unsqueeze(0).to(dev)
            if out.shape != gt.shape:
                out = F.interpolate(out, size=gt.shape[2:], mode="bilinear", align_corners=False)
            with torch.no_grad():
                rows.append([t["filter"], t["target"], t["reference"], float(benchmark.calculate_psnr(out, gt)),
                             float(benchmark.calculate_ssim(out, gt)), lp(out * 2 - 1, gt * 2 - 1).item(), secs])
            if (n + 1) % 100 == 0:
                print(f"steps={steps} {n + 1}/{len(T)}  {(time.time() - t0) / (n + 1):.1f} s/task", flush=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["filter", "target", "reference", "psnr", "ssim", "lpips", "tto_s"])
            w.writerows(rows)
        arr = np.array([r[3:] for r in rows])
        summary[steps] = dict(n=len(rows), psnr=arr[:, 0].mean(), ssim=arr[:, 1].mean(), lpips=arr[:, 2].mean())
        print(f"steps={steps}: {summary[steps]}", flush=True)
    json.dump(summary, open(os.path.join(a.out_dir, "metadc_iffi_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iffi_root", default=iffi_tasks.IFFI_ROOT, required=iffi_tasks.IFFI_ROOT is None)
    ap.add_argument("--meta_weights", default=os.path.join(REPO, "weights/meta_model_ft.pth"))
    ap.add_argument("--steps_list", default="10,100")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--window_size", type=int, default=13)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only; 0 = all 1408 tasks")
    main(ap.parse_args())
