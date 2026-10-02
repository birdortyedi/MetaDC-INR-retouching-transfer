#!/bin/bash
# Component ablation (Table 3): meta-train one variant, then evaluate it at 100 TTO steps under the
# Table 1 protocol (batch 512, all 1,342 RTD benchmark tasks).
#
#   bash scripts_benchmarking/run_component_ablation.sh <RTD root> <variant> [gpu]
#
#   <variant>  no_global | no_local | naive_coords | single_scale | concat_cond   (see model_ablate.py)
#              none = the full architecture, retrained with the same script (control)
#
# Checkpoints go to weights/meta_ablate_<variant>.pth (git-ignored); if one exists, training is skipped.
# benchmark.py writes FINAL_BENCHMARK_RESULTS_multiscale.txt into results/ablation/<variant>/.
set -euo pipefail
RTD=$1; V=$2; GPU=${3:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$REPO/results/ablation/$V
mkdir -p "$OUT" "$REPO/weights"
FLAG=(); [ "$V" = none ] || FLAG=(--ablate "$V")
CK=$REPO/weights/meta_ablate_$V.pth

if [ ! -f "$CK" ]; then
  (cd "$OUT" && python "$REPO/meta_train.py" --dataset_path "$RTD" --gpu "$GPU" --out_prefix "meta_ablate_$V" "${FLAG[@]}")
  mv "$OUT/meta_ablate_${V}_final.pth" "$CK"
fi

# benchmark.py would silently fall back to random initialization if the weights did not fit the variant,
# so the checkpoint is strict-loaded into exactly the class benchmark.py builds before evaluating.
python - "$REPO" "$CK" "$V" <<'EOF'
import sys, torch
repo, ck, v = sys.argv[1:]
sys.path.insert(0, repo)
from model import InRetouchNR
from model_ablate import InRetouchNRAblate
m = InRetouchNR(hidden_dim=128) if v == "none" else InRetouchNRAblate(hidden_dim=128, ablate=v)
m.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)
print(f"checkpoint OK: {ck} ({sum(p.numel() for p in m.parameters()):,} parameters)")
EOF

(cd "$OUT" && python "$REPO/scripts_benchmarking/benchmark.py" --dataset_path "$RTD" --meta_weights "$CK" \
    --steps_list 100 --batch_size 512 --gpu "$GPU" "${FLAG[@]}")
