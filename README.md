# Rapid 2D Magnetotelluric Forward Modeling via Automatic Differentiation Fourier Neural Operato with Gradient Supervision

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning framework that combines **Fourier Neural Operators (FNO)**, **Deep Operator Networks (DeepONet)**, and **physics-guided attention mechanisms** to act as a fast surrogate model for 2D magnetotelluric (MT) forward modeling.

## Overview

Magnetotelluric forward modeling — computing electromagnetic field responses from subsurface conductivity structures — is computationally expensive with traditional finite-difference methods. ADFNO learns the nonlinear mapping from 2D conductivity distributions to MT responses (apparent resistivity and phase), achieving orders-of-magnitude speedup while maintaining high accuracy.

### Key Features

- **Hybrid architecture**: FNO performs frequency-domain global feature extraction; DeepONet branch/trunk networks encode spatial coordinates and conductivity distributions; UFNO blocks with simplified U-Nets refine local features
- ** dual attention**: Frequency attention adaptively weights contributions across frequencies; spatial attention uses conductivity gradient information to guide feature learning
- **Multi-task learning**: Simultaneously predicts both electromagnetic field gradients and surface forward responses with an automatic weighting scheme
- **Multi-frequency generalization**: Trained on multiple frequencies simultaneously, generalizes across frequencies without retraining

## Architecture

```
Input (σ, grid, freq)
    │
    ├── Trunk (DeepONet): spatial encoding + spatial attention
    ├── Branch (DeepONet): conductivity encoding + frequency attention
    │
    ├── Feature Fusion (element-wise product)
    │
    ├── UFNO Blocks (×N): Fourier layers + simplified U-Net
    │
    ├── Output Head 1: EM field gradients → pred_grad (B, n_freq, dy-1, dz-1, 4)
    └── Output Head 2: Forward responses → pred_y   (B, n_freq, dy, 4)
```

## Project Structure

```
.
├── src/
│   ├── Train.py            # Training entry point
│   ├── Predict.py          # Inference/prediction script
│   ├── config.yml          # Model and training configuration
│   └── evaluate.ipynb      # Evaluation and visualization notebook
├── utils/
│   ├── ADFNO.py            # Core ADFNO model definition
│   ├── FNO.py              # Baseline FNO-2D implementation
│   ├── KAN.py              # Kolmogorov-Arnold Network module
│   ├── derivative.py       # MT physics: PDE residuals, EM field computation, 1D reference solutions
│   └── load_data.py        # Data loading and normalization
├── Data/
│   ├── MT2D_secondary_direct.py  # 2D MT finite-difference forward solver (ground truth)
│   ├── gaussian_random_fields.py # Synthetic conductivity model generation
│   ├── main_forward.ipynb        # Forward modeling data generation notebook
│   └── datasets/                 # Training/test datasets (.npz)
├── model/                   # Saved model checkpoints
├── results/                 # Prediction outputs and metrics
├── Log/                     # Training logs
└── temp/                    # Temporary model snapshots
```

## Installation

```bash
# Clone the repository
git clone https://github.com/2323948941/adfno.git

# Install dependencies
pip install torch numpy scipy pyyaml matplotlib
```

## Usage

### Training

```bash
python src/Train.py grid_30000
```

The `grid_30000` argument refers to the configuration section in `src/config.yml`. You can modify parameters there or add new configuration sections.

### Inference

```bash
# With explicit model path
python src/Predict.py grid_30000 path/to/model.pkl

# Auto-find model in temp/ directory
python src/Predict.py grid_30000
```

### Configuration

Key parameters in `src/config.yml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `modes` | Number of truncated Fourier modes | 18 |
| `width` | Feature channel width after lifting | 32 |
| `layer_num` | Number of Fourier layers | 4 |
| `f_idx` | Frequency index range | [1, 16, 1, 16] |
| `n_train` / `n_test` | Number of train/test samples | 3000 / 300 |
| `tf_lr` | Initial learning rate | 0.001 |
| `patience` | Early stopping patience | 20 |

## Data Generation

The project uses synthetic 2D conductivity models generated via Gaussian random fields. Forward responses (apparent resistivity, phase) are computed using a second-order finite-difference solver with the secondary field method (both TE and TM modes).

```bash
# Generate datasets using the provided notebook
jupyter notebook Data/main_forward.ipynb
```

## Output Channels

The model predicts 4 output channels for MT forward responses:
- `rhoxy` — Apparent resistivity (TE mode, xy-polarization)
- `phsxy` — Phase (TE mode, xy-polarization)
- `rhoyx` — Apparent resistivity (TM mode, yx-polarization)
- `phsyx` — Phase (TM mode, yx-polarization)



## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project builds upon the [Fourier Neural Operator (FNO)](https://github.com/neuraloperator/neuraloperator) framework
- Gaussian random field generation adapted from [Bruno Sciolla's implementation](https://github.com/bsciolla/gaussian-random-fields)
- 1D MT analytical solutions based on Wait (1953)
