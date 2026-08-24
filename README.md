# SENK: SO(3) Equivariant Neural Kalman Networks

**Official PyTorch implementation of**  
*"Response-state learning for transferable vibrational spectroscopic characterization with electronic priors."*

SENK is a response-state framework for transferable vibrational spectroscopy. It combines an SO(3)-equivariant backbone for Hessian, dipole-derivative, and polarizability-derivative prediction with an Equivariant Neural Kalman (ENK) module for state-dependent refinement and reliability sensing, and an NBO-informed electronic-prior pathway for consistency regularization and bounded spectral calibration.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Data Preparation](#data-preparation)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Training](#training)
- [Inference](#inference)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Given atomic numbers and equilibrium Cartesian coordinates, SENK predicts four spectroscopically relevant response quantities:

- H<sub>ii</sub> — diagonal Hessian block
- H<sub>ij</sub> — off-diagonal Hessian block
- ∂μ/∂R — dipole derivative
- ∂α/∂R — polarizability derivative

These quantities are passed to a harmonic normal-mode analysis pipeline to generate IR and Raman spectra.

SENK contains four main components:

| Component | Role |
|---|---|
| **EquiformerV2 backbone** | SO(3)-equivariant representation learning for spectroscopic response tensors. |
| **ENK** | State-dependent refinement of selected equivariant channels and atom-resolved reliability sensing. |
| **Electronic prior / CR** | NBO-informed electronic features used for representation conditioning and training-time consistency regularization. |
| **GSC** | Bounded, branch-specific calibration applied at inference when the electronic prior supports a correction. |

### Benchmark summary

Mean absolute errors reported in the accompanying manuscript:

| Dataset | Branch | DetaNet | SENK (ENK off) | SENK (ENK on) |
|---|---|---:|---:|---:|
| QM9S | ∂μ/∂R | 0.0182 | 0.0091 | **0.0081** |
| QM9S | ∂α/∂R | 0.2012 | 0.0780 | **0.0775** |
| QM9S | H | 0.0822 | 0.0267 | **0.0262** |
| QMe14S | ∂μ/∂R | 0.0395 | 0.0117 | **0.0106** |
| QMe14S | ∂α/∂R | 0.4673 | 0.1163 | **0.1121** |
| QMe14S | H | 0.1074 | 0.0303 | **0.0279** |

See the manuscript for the full benchmark definitions, ablations, spectral comparisons, and external validation.

---

## Architecture

### EquiformerV2 adaptation

SENK adapts EquiformerV2 to predict spectroscopic response tensors directly from molecular geometry. The Hessian uses separate derivative routes for H<sub>ii</sub> and H<sub>ij</sub>, while the polarizability branch retains rank-2 equivariant information in the readout.

### ENK bridge

ENK is an equivariance-preserving confidence filter applied to selected SO(3) latent channels before tensor readout. The learned Kalman gain is scalar over magnetic components, preserving rotational equivariance while providing a response-state reliability signal.

### Electronic prior

The electronic-prior model follows EMPP design principles and is trained with SIMG and qcMol supervision. Geometry-derived NBO features are mapped to atom- and edge-level prior representations that condition the SENK backbone and support branch-specific consistency regularization.

### Guided Spectral Calibration

GSC uses training-set electronic statistics to apply bounded corrections to eligible response branches at inference time. The calibration is designed to preserve response states that are already well described by the backbone.

---

## Repository Structure

```text
SENK_main_4.10/
├── train_senk.py                  # Main SENK training entry point
├── v2_spectra_infer.py            # IR/Raman inference and plotting
├── dataloader_spectra.py          # QM9S/QMe14S data loaders
├── prep_prior_cache.py            # Electronic-prior cache generation
├── compute_nbo_train_stats.py     # GSC training-statistics generation
├── nbo_consistency_loss.py        # Electronic-prior consistency losses
├── nbo_spectral_calibration.py    # Guided spectral calibration
├── nets/                          # SENK and EquiformerV2 model components
├── nbo_nets/                      # NBO electronic-prior model and training code
├── detanet_nets/                  # DetaNet baseline implementation and weights
├── datasets/                      # Dataset loaders and preprocessing scripts
├── tools/                         # Skeleton-building and auxiliary utilities
└── requirements.txt
```

---

## Installation

SENK has been tested on Linux with Python 3.10 and PyTorch 2.2.2 with CUDA 12.1. We recommend using `conda`.

```bash
conda create -n senk python=3.10 -y
conda activate senk
```

Install PyTorch and PyTorch Geometric dependencies:

```bash
pip install torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

pip install torch-scatter torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

pip install torch-geometric==2.5.3
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_geometric, torch_scatter, e3nn; print('PyG+e3nn OK')"
python -c "from nets.clean_equiformer_polar_ext import CleanEquiformerPolarExt; print('SENK OK')"
```

---

## Datasets

SENK uses four main data sources:

| Dataset | Use |
|---|---|
| **QM9S** | Supervision for Hessian, dipole-derivative, and polarizability-derivative prediction. |
| **QMe14S** | Main spectroscopy dataset covering a broader 14-element chemical space. |
| **SIMG** | Heterogeneous NBO graph supervision for electronic-prior pretraining. |
| **qcMol** | Atom- and bond-resolved quantum-chemical supervision for electronic-prior adaptation. |

Raw QM9S, QMe14S, SIMG, and qcMol files are not bundled with this repository. Please obtain them from the original sources cited in the manuscript and use the preprocessing utilities provided here.

---

## Data Preparation

All commands below assume the repository root as the working directory.

### 1. QMe14S preprocessing

Extract the raw `OPT_186102.h5` file:

```bash
python datasets/QMe14S/qme14s_opt186102_extract.py \
    --h5_path datasets/QMe14S/OPT_186102.h5 \
    --output datasets/QMe14S/qme14s_opt186102_raw.pt
```

Convert it to the training format:

```bash
python datasets/QMe14S/qme14s_opt186102_preprocess.py \
    --input datasets/QMe14S/qme14s_opt186102_raw.pt \
    --output datasets/QMe14S/qme14s_opt186102.pt
```

Optional flags include `--max_items` for quick tests and `--max_atoms` for atom-count filtering.

### 2. Skeleton generation

Generate the structural skeleton required by the electronic-prior model.

QM9S:

```bash
python tools/build_qm9s_skeletons.py \
    --qm9s datasets/qm9s.pt \
    --output_dir datasets/new_skeleton \
    --lp_ckpt tools/lp_pred_model.ckpt \
    --lp_device cuda
```

QMe14S:

```bash
python tools/build_qm9s_skeletons.py \
    --qm9s datasets/QMe14S/qme14s_opt186102.pt \
    --output_dir datasets/QMe14S \
    --lp_ckpt tools/lp_pred_model.ckpt \
    --lp_device cuda
```

### 3. Electronic-prior cache

For EP-enabled training modes, precompute the electronic-prior cache:

```bash
python prep_prior_cache.py \
    --dataset qme14s_opt186102 \
    --task spectra4 \
    --electron-prior-mode simg \
    --electron-prior-ckpt nbo_nets/checkpoints/2025-11-10_11-06-07_nbo_model_with_links/nbo_model_with_links_best.pt \
    --electron-prior-stats nbo_nets/checkpoints/2025-11-10_11-06-07_nbo_model_with_links/nbo_model_with_links_norm_stats.pt \
    --shard-size 256 \
    --keep-shards 2 \
    --gpu 0
```

Non-EP modes (`equiformer_v2`, `equiformer_v2_enk`) can skip this step.

---

## Pretrained Checkpoints

The repository includes the following auxiliary checkpoints:

| Checkpoint | Path | Purpose |
|---|---|---|
| NBO predictor | `nbo_nets/checkpoints/2025-11-10_11-06-07_nbo_model_with_links/nbo_model_with_links_best.pt` | Electronic-prior inference |
| NBO normalization statistics | `nbo_nets/checkpoints/2025-11-10_11-06-07_nbo_model_with_links/nbo_model_with_links_norm_stats.pt` | Matching normalization statistics |
| Lone-pair predictor | `tools/lp_pred_model.ckpt` | Skeleton construction |
| DetaNet QM9S weights | `detanet_nets/qm9spectra/*.pth` | Baseline comparison |

The main SENK response-state checkpoints are released alongside the manuscript. See the manuscript's data and code availability statement for release information.

---

## Training

`train_senk.py` supports the following model configurations:

| Mode | Backbone | ENK | EP |
|---|---|---|---|
| `clean_equiformer` | V1 Equiformer | — | — |
| `equiformer_electron_prior` | V1 Equiformer | — | ✓ |
| `equiformer_v2` | EquiformerV2 | — | — |
| `equiformer_v2_enk` | EquiformerV2 | ✓ | — |
| `equiformer_v2_enk_ep` | EquiformerV2 | ✓ | ✓ |

Example: train full SENK on QMe14S.

```bash
python train_senk.py \
    --mode equiformer_v2_enk_ep \
    --dataset qme14s_opt186102 \
    --task polar \
    --batch_size 32 \
    --epochs 300 \
    --lr 2e-4 \
    --weight_decay 1e-2 \
    --v2_num_layers 8 \
    --v2_num_gaussians 64 \
    --v2_max_num_neighbors 64 \
    --v2_radius 5.0 \
    --enk_enabled \
    --electron_prior_config configs/electron_prior_default.yaml \
    --output_dir outputs/senk_v2_enk_ep_qme14s_polar
```

Supported branch-specific tasks include `polar`, `depolar`, `sobolev_polar`, `dipole`, `dedipole`, `hii`, `hij`, and `spectra4`.

### Training the electronic-prior model

```bash
# Stage 1: SIMG pretraining
python nbo_nets/train_nbo_v2.py --config configs/nbo_simg.yaml

# Stage 2/3: SIMG + qcMol joint training
python nbo_nets/train_nbo_joint.py --config configs/nbo_joint.yaml
```

### GSC statistics

After training the required ENK+EP branches, compute the GSC statistics once for the training set:

```bash
python compute_nbo_train_stats.py \
    --hij_ckpt outputs/senk_v2_enk_ep_qme14s_hij/best.pt \
    --dd_ckpt  outputs/senk_v2_enk_ep_qme14s_dd/best.pt \
    --dp_ckpt  outputs/senk_v2_enk_ep_qme14s_dp/best.pt \
    --output   outputs/nbo_train_stats.pt
```

---

## Inference

`v2_spectra_infer.py` loads the four SENK response checkpoints (H<sub>ii</sub>, H<sub>ij</sub>, dipole derivative, and polarizability derivative) and generates IR/Raman spectra with diagnostic plots.

Recommended SENK-EP inference:

```bash
python v2_spectra_infer.py \
    --xyz examples/molecule.xyz \
    --ep \
    --gpu 0 \
    --out_png outputs/spectrum.png
```

ENK on/off comparison:

```bash
python v2_spectra_infer.py \
    --xyz examples/molecule.xyz \
    --gpu 0 \
    --out_png outputs/spectrum.png
```

DetaNet comparison:

```bash
python v2_spectra_infer.py \
    --xyz examples/molecule.xyz \
    --compare_detanet \
    --gpu 0 \
    --out_png outputs/spectrum.png
```

---

## Citation

Citation information will be added once a preprint or published version of the manuscript is available.

**Authors:** Zetong Li, Zhuosong Xie, Hengyu Fan, Jiaao Yu, Juanni Wu, Honglin Li

---

## Acknowledgements

SENK builds on several open-source projects and datasets, including:

- **EquiformerV2** — Liao et al., ICLR 2024
- **DetaNet** — Zou et al., *Nature Computational Science* 3, 957–964 (2023)
- **SIMG** — Boiko et al., *Nature Machine Intelligence* 7, 771–781 (2025)
- **EMPP** — An et al., arXiv:2502.08209 (2025)
- **e3nn**
- **PyTorch Geometric**
- **RDKit**
- **QM9S, QMe14S, SIMG, and qcMol** datasets

We thank the developers and dataset contributors for making these resources available. Funding information and additional acknowledgements are provided in the manuscript.

---

## License

This project is released under the **MIT License**. Vendored components under `nets/eqv2_core/` (EquiformerV2) and `detanet_nets/` (DetaNet) retain their original upstream licenses, as indicated in the corresponding source files.
