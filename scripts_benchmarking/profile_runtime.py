#!/usr/bin/env python
"""End-to-end runtime and peak-memory profile of MetaDC-INR (Table 5). Measurement only.

Times benchmark.py's own run_tto, imported and called UNCHANGED (Table 1 protocol: all parameters adapted, Adam at
lr 1e-3, batch 512, window 13), on real RTD benchmark images at their native resolution. Stages per task:
  prep    loading the three images, building the model, loading the weights  (= total - tto - render)
  tto     the optimization loop, as returned by run_tto (synchronized at both ends)
  render  the final full-resolution forward pass on the target image
  total   the whole run_tto call, synchronized before and after
The render time is read by a subclass of InRetouchNR that times its forward only in eval mode; the parameters and
the computation are identical, so the state dict loads strictly. Peak memory = torch.cuda.max_memory_allocated,
reset before each task. Task 0 is a warm-up and is excluded from the means.

Tasks: Preset_146 x all 61 targets, references from references_file.txt.

    python scripts_benchmarking/profile_runtime.py --rtd <RTD> --steps 100 --out_dir results/profile
    python scripts_benchmarking/profile_runtime.py --rtd <RTD> --steps 100 --hidden_dim 64 --out_dir results/profile
The d=64 row is timing only (randomly initialized weights of the d=64 architecture).
"""
import argparse, csv, importlib.util, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, REPO)
import numpy as np, torch, torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

spec = importlib.util.spec_from_file_location("benchmark", os.path.join(REPO, "scripts_benchmarking", "benchmark.py"))
benchmark = importlib.util.module_from_spec(spec); spec.loader.exec_module(benchmark)
_Base = benchmark.InRetouchNR
_render = []


class TimedRender(_Base):
    def forward(self, *a, **k):
        if self.training:
            return super().forward(*a, **k)
        torch.cuda.synchronize(); t = time.perf_counter()
        out = super().forward(*a, **k)
        torch.cuda.synchronize(); _render.append(time.perf_counter() - t)
        return out


benchmark.InRetouchNR = TimedRender


def tasks(rtd, preset="Preset_146"):
    b = os.path.join(rtd, "Benchmark")
    refs = [l.strip().split(",") for l in open(os.path.join(b, "references_file.txt")) if l.strip()]
    return [dict(t_nat=os.path.join(b, "Test", "natural", t), r_nat=os.path.join(b, "Test_References", "natural", r),
                 r_ret=os.path.join(b, "Test_References", "Presets", preset, r), gt=os.path.join(b, "Test", "Presets", preset, t),
                 target=t) for t, r in sorted(refs)]


def main(a):
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev); torch.zeros(1, device=dev); torch.cuda.synchronize(dev)   # initialise CUDA before the memory counters are reset
    weights = a.meta_weights if a.hidden_dim == 128 else "__random_init__"   # d=64: no checkpoint; timing only
    T = tasks(a.rtd); tt = transforms.ToTensor()
    assert len(T) == 61, len(T)
    if a.limit > 0:
        T = T[:a.limit]   # smoke tests only
    rows = []
    for n, t in enumerate(T):
        _render.clear(); torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out, tto = benchmark.run_tto(weights, t["t_nat"], t["r_nat"], t["r_ret"], dev, steps=a.steps,
                                     batch_size=512, window_size=13, hidden_dim=a.hidden_dim)
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated(dev) / 2**20
        W, H = Image.open(t["t_nat"]).size
        psnr = float("nan")
        if a.hidden_dim == 128:
            gt = tt(Image.open(t["gt"]).convert("RGB")).unsqueeze(0).to(dev)
            if out.shape != gt.shape:
                out = F.interpolate(out, size=gt.shape[2:], mode="bilinear", align_corners=False)
            psnr = float(benchmark.calculate_psnr(out, gt))
        assert len(_render) == 1, f"expected one eval forward, got {len(_render)}"
        render = _render[0]
        rows.append([a.hidden_dim, a.steps, n, t["target"], W, H, total - tto - render, tto, render, total, mem, psnr])
    path = os.path.join(a.out_dir, f"profile_metadc_d{a.hidden_dim}_s{a.steps}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["hidden_dim", "steps", "task", "target", "W", "H", "prep_s", "tto_s", "render_s", "total_s", "peak_mib", "psnr"])
        w.writerows(rows)
    r = np.array([x[6:11] for x in rows[1:]], dtype=float)   # exclude warm-up task
    print(f"d={a.hidden_dim} steps={a.steps} n={len(r)} prep={r[:,0].mean():.3f}s tto={r[:,1].mean():.2f}s "
          f"({1000*r[:,1].mean()/a.steps:.1f} ms/step) render={r[:,2].mean():.3f}s total={r[:,3].mean():.2f}s "
          f"peak={r[:,4].max():.0f} MiB  mean MP={np.mean([x[4]*x[5] for x in rows])/1e6:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtd", required=True)
    ap.add_argument("--meta_weights", default=os.path.join(REPO, "weights/meta_model_ft.pth"))
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
