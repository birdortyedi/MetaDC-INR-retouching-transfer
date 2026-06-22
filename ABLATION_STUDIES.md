# MetaDC-INR Ablation and Benchmarking Scripts Specification

This document specifies the official ablation and benchmarking scripts for **MetaDC-INR** (excluding Lie-algebra based variants). These scripts evaluate the model's core components: the meta-learned prior, baseline comparisons, TTO strategies, and robustness to sampling randomness.

---

## 1. Convergence Ablation: Meta-Learned Prior vs. Random Init

- **Script Path**: `scripts_benchmarking/profile_convergence.py`
- **Description**: Profiles the convergence rate of MetaDC-INR by comparing TTO convergence trajectories starting from the meta-learned initialization (prior) versus starting from scratch (random initialization).
- **Execution Command**:
  ```bash
  python scripts_benchmarking/profile_convergence.py \
      --dataset_path /home/birdortyedi/PycharmProjects/shadow-aware-l0-smoothing/datasets/Retouch_Transfer_Dataset \
      --meta_weights weights/meta_model_ft.pth \
      --num_samples 5 \
      --steps 200 \
      --gpu 0
  ```
- **Outputs**:
  - Plots average convergence curves (PSNR, SSIM, ΔE) to `convergence_profiles/convergence_plot.png`.
  - Saves raw step-by-step metrics to `convergence_profiles/convergence_data.json`.

---

## 2. Adaptation Efficiency: Meta-Initialization vs. Pre-trained Baselines

This ablation compares MetaDC-INR's Reptile-based initialization against models pre-trained using standard supervised learning with different optimizers (SGD, Adam, AdamW).

### Part A: Train Baselines
- **Script Path**: `scripts_training/train_baseline.py`
- **Description**: Pre-trains baseline models on the supervised dataset using SGD, Adam, or AdamW.
- **Execution Command**:
  ```bash
  python scripts_training/train_baseline.py \
      --dataset_path /home/birdortyedi/PycharmProjects/shadow-aware-l0-smoothing/datasets/Retouch_Transfer_Dataset \
      --optimizer adamw \
      --epochs 5 \
      --gpu 0
  ```
- **Outputs**: Baseline weights saved to `weights/adamw_pretrained_model.pth`.

### Part B: Benchmarking Adaptation
- **Script Path**: `scripts_benchmarking/ablation_meta_vs_baselines.py`
- **Description**: Runs TTO starting from the meta-initialization and each baseline weight file, then compares adaptation rate and final PSNR.
- **Execution Command**:
  ```bash
  python scripts_benchmarking/ablation_meta_vs_baselines.py \
      --dataset_path /home/birdortyedi/PycharmProjects/shadow-aware-l0-smoothing/datasets/Retouch_Transfer_Dataset \
      --meta_weights weights/meta_model_ft.pth \
      --num_samples 10 \
      --gpu 0
  ```
- **Outputs**:
  - Comparison plot saved to `ablation_meta_vs_baselines.png`.
  - Numerical summary printed to stdout.

---

## 3. TTO Strategy Ablation: Full vs. Selective vs. Head-Only

- **Script Path**: `scripts_analysis/selective_tto_analysis.py`
- **Description**: Evaluates which parts of the MetaDC-INR architecture should be adapted during TTO. It compares:
  1. `full`: Fine-tuning all parameters.
  2. `selective`: Freezing the CNN branches and updating only the INR layers.
  3. `head_only`: Freezing all layers except the dual-path output projection.
- **Execution Command**:
  ```bash
  python scripts_analysis/selective_tto_analysis.py \
      --dataset_path /home/birdortyedi/PycharmProjects/shadow-aware-l0-smoothing/datasets/Retouch_Transfer_Dataset \
      --meta_weights weights/meta_model_ft.pth \
      --steps 20 \
      --num_samples 5 \
      --gpu 0
  ```
- **Outputs**: Prints comparison of average PSNR across strategies to stdout.

---

## 4. TTO Robustness Ablation: Multi-Seed Variance

- **Script Path**: `scripts_benchmarking/measure_variance.py`
- **Description**: Measures the sensitivity of MetaDC-INR's TTO performance to the randomized sub-pixel coordinates. Runs TTO across 5 different seeds (`[42, 100, 2026, 999, 12345]`) on all benchmark tasks.
- **Execution Command**:
  ```bash
  python scripts_benchmarking/measure_variance.py \
      --dataset_path /home/birdortyedi/PycharmProjects/shadow-aware-l0-smoothing/datasets/Retouch_Transfer_Dataset \
      --meta_weights weights/meta_model_ft.pth \
      --gpu 0
  ```
- **Outputs**:
  - Incremental progress saved to `logs_and_reports/variance_results_temp.json`.
  - Consolidated raw data saved to `logs_and_reports/variance_results.json`.
  - Detailed summary tables (Mean ± Std, Min/Max per seed) printed and saved to `logs_and_reports/variance_results.txt`.

---

## 5. Dual-Path Disentanglement Visualization

- **Script Path**: `scripts_visualization/visualize_decomposition.py`
- **Description**: Decomposes the final retouching output into its Affinity Matrix (local color mapping) and Detail Residual (high-frequency refinements) components to verify that spatial and tonal adjustments are cleanly disentangled.
- **Execution Command**:
  ```bash
  python scripts_visualization/visualize_decomposition.py \
      --dataset_path /home/birdortyedi/PycharmProjects/shadow-aware-l0-smoothing/datasets/Retouch_Transfer_Dataset \
      --meta_weights weights/meta_model_ft.pth \
      --gpu 0
  ```
- **Outputs**: Saves decomposition visual strips (Original, Matrix component, Detail component, Fused result) to `visualization_outputs/`.
