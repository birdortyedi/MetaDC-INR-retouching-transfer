# MetaDC-INR: Task-Adaptive Meta-Learning for Disentangled Contextual Retouching Transfer

## 1. Overview

Our solution addresses deterministic retouching transfer through disentanglement of contextual information and meta-learning. The proposed architecture, **MetaDC-INR**, is designed to disentangle contextual style descriptors from spatially-continuous structural adjustments, ensuring retouching is applied as a precise coordinate-based transformation rather than a stochastic pixel generation. MetaDC-INR leverages a hybrid framework where an Implicit Neural Representation (INR) is dynamically modulated by multi-scale features, initialized via a task-adaptive meta-initialization strategy to facilitate better convergence during test-time optimization (TTO).

## 2. Model Architecture

### 2.1. Dual-Branch Feature Extraction

To capture high-level aesthetic priors without compromising local fidelity, we employ a dual-branch feature extractor:

*   **Local Context Branch**: A 4-layer dilated convolutional network with a cumulative receptive field of **29×29**, employing dilation rates of [2, 4, 8] to aggregate multi-scale spatial interactions. A Squeeze-and-Excitation (SE) block [1] performs channel-wise recalibration to prioritize salient color and luminance ranges.
*   **Global Context Branch**: Extracts a latent "mood" embedding through adaptive average pooling to capture the holistic tonal distribution of the input.

### 2.2. Spatially-Continuous MLP (INR)

The core of MetaDC-INR is a spatially-continuous MLP that maps hybrid coordinates to retouching parameters. The coordinate space is constructed from three encodings:

1.  **Relative Positional Encoding**: High-frequency positional encoding (10 frequencies) for sub-pixel offsets.
2.  **Integrated Positional Encoding (IPE)** [2]: Low-frequency encoding (3 frequencies) with a 0.02 blurring factor for dampened absolute spatial awareness.
3.  **Color Encoding**: 8-frequency encoding of input RGB values for localized tone mapping.

These features are fused and injected into the MLP via **Feature-wise Linear Modulation (FiLM)** [3], where conditioning parameters dynamically scale and shift the latent structure features.

### 2.3. Dual-Path Output Head

To ensure stable optimization, MetaDC-INR decouples adjustments into two specialized output paths:

1.  **Affinity Matrix Path**: Predicts a **3×4 local color transformation matrix** ($M$) per pixel for operations like exposure correction and white balance.
2.  **Detail Residual Path**: Predicts an **RGB delta** ($\Delta$), constrained by a scaled `tanh` function, for high-frequency texture and edge refinements.
3.  **Fusion**: $Y_{i,j} = (M_{i,j} \times X_{i,j}) + \Delta_{i,j}$, with heads initialized as an identity map to provide an unbiased starting point for task adaptation.

## 3. Training and Optimization Strategy

### 3.1. Meta-Learning (Reptile)

A critical component of MetaDC-INR is the application of the **Reptile** meta-learning algorithm [4] to establish a task-agnostic weight initialization, enabling rapid adaptation during a subsequent 500-step TTO loop at inference.

### 3.2. Sub-Pixel Sampling

To ensure spatial continuity and prevent aliasing artifacts, TTO utilizes an advanced sub-pixel sampling strategy:

*   **Sharp Context Patches**: 41×41 patches (nearest-neighbor) condition the convolutional branches.
*   **Smooth Patches**: 13×13 patches (bilinear) operate within the INR backbone.

### 3.3. Loss Functions

The test-time adaptation is governed by a weighted composite objective:

$$\mathcal{L} = \mathcal{L}_{\text{Charb}} + \lambda_{S}\mathcal{L}_{\text{SSIM}} + \lambda_{L}\mathcal{L}_{\text{Lab}} + \lambda_{T}\mathcal{L}_{\text{TV}}$$

*   **Charbonnier Loss** ($\mathcal{L}_{\text{Charb}}$): Smooth L1 with $\beta = 0.01$ for stable gradients near zero error.
*   **SSIM Loss** ($\mathcal{L}_{\text{SSIM}}$, $\lambda_S = 0.2$): Enforces perceptual and structural fidelity.
*   **Lab-Color Loss** ($\mathcal{L}_{\text{Lab}}$, $\lambda_L = 0.05$): Minimizes Euclidean distance in Lab color space for perceptual accuracy.
*   **Total Variation Loss** ($\mathcal{L}_{\text{TV}}$, $\lambda_T = 0.001$): Promotes local smoothness to suppress latent grid noise.

## 4. Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Hidden Dimension | 128 |
| Receptive Field | 29×29 |
| Context Window (Sharp) | 41×41 |
| INR Patch (Smooth) | 13×13 |
| Meta Batch Size | 1024 |
| Inner Steps (Train) | 12 |
| TTO Steps (Inference) | 500 |
| Optimizer | Adam |
| Meta LR (Outer) | 0.05 (decaying) |
| Inner LR | 1e-3 |

## 5. References

1.  J. Hu, L. Shen, and G. Sun. "Squeeze-and-Excitation Networks." *CVPR*, 2018.
2.  J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan. "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields." *ICCV*, 2021.
3.  E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville. "FiLM: Visual Reasoning with a General Conditioning Layer." *AAAI*, 2018.
4.  A. Nichol, J. Achiam, and J. Schulman. "On First-Order Meta-Learning Algorithms." *arXiv:1803.02999*, 2018.
