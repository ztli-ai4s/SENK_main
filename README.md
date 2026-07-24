# SENK: SO(3) Equivariant Neural Kalman Networks

**Official PyTorch implementation of**
*"Response-state learning for vibrational spectra of molecules with stereoelectronic effects."*

SENK is a response-state cascade that combines an equivariant transformer backbone for Hessian, dipole-derivative and polarizability-derivative learning, an Equivariant Neural Kalman (ENK) bridge for state-dependent refinement and reliability sensing, and an NBO-informed Electronic Prior (EP) pathway coupling Consistency Regularization (CR) with bounded, branch-specific Guided Spectral Calibration (GSC). SENK outperforms DetaNet on QM9S and QMe14S while preserving full-spectrum IR and Raman fidelity from small molecules to drug-like and biomolecular systems.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Data Preprocessing](#data-preprocessing)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Training](#training)
- [Inference](#inference)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Vibrational spectral prediction becomes inaccurate when localized stereoelectronic environments perturb intermediate response states and high-risk response units dominate characteristic spectral fingerprints. SENK treats response prediction as a staged state-estimation and calibration problem rather than a single end-to-end regression.

Given atomic numbers and equilibrium Cartesian coordinates, SENK predicts four spectroscopically relevant per-atom and per-edge quantities:

- H<sub>ii</sub> — diagonal Hessian matrix block
- H<sub>ij</sub> — off-diagonal Hessian matrix block
- ∂μ/∂R — dipole derivative (IR intensity)
- ∂α/∂R — polarizability derivative (Raman intensity)

These four response states are subsequently passed to a harmonic normal-mode analysis pipeline to generate IR absorption and Raman scattering spectra.

SENK introduces three coupled modules on top of an SO(3)-equivariant EquiformerV2 backbone:

| Module | Role |
|---|---|
| **ENK** (Equivariant Neural Kalman) | Environment-dependent confidence update on selected equivariant channels before tensor readout; supplies an atom-resolved reliability signal. |
| **EP** (Electron Prior) | NBO-informed electronic prior injected at the EquiformerV2 transformer block (Point A) and used to modulate ENK observation noise via FiLM conditioning. |
| **GSC** (Guided Spectral Calibration) | Bounded, branch-specific post-hoc calibration guided by NBO-derived training-set statistics. |
| **CR** (Consistency Regularization) | Training-time regularization enforcing the consensus-weighted Badger-type relation between H<sup>ℓ</sup> and NBO bond occupancy, the acoustic sum rule for Born charges, and delocalization–polarizability consistency. |

The overall training objective for the EP pathway is

<div align="center">
<b>ℒ<sub>EP</sub>(θ) = ℒ<sub>sup</sub>(θ) + λ<sub>CR</sub> · ℒ<sub>CR</sub>(θ)</b>
</div>

where ℒ<sub>CR</sub> aggregates the Badger-type, acoustic-sum-rule, dipole-derivative and polarizability-derivative consistency terms.

### Key results

Mean absolute errors on the QM9S and QMe14S benchmarks (full-spectrum reference supervision; values taken from the accompanying manuscript):

| Dataset | Branch | DetaNet | SENK (ENK off) | SENK (ENK on) |
|---|---|---:|---:|---:|
| QM9S | ∂μ/∂R (dipole derivative) | 0.0182 | 0.0091 | **0.0081** |
| QM9S | ∂α/∂R (polarizability derivative) | 0.2012 | 0.0780 | **0.0775** |
| QM9S | H (Hessian) | 0.0822 | 0.0267 | **0.0262** |
| QMe14S | ∂μ/∂R (dipole derivative) | 0.0395 | 0.0117 | **0.0106** |
| QMe14S | ∂α/∂R (polarizability derivative) | 0.4673 | 0.1163 | **0.1121** |
| QMe14S | H (Hessian) | 0.1074 | 0.0303 | **0.0279** |

The equivariant backbone provides the dominant accuracy gain, while ENK contributes a smaller but consistent refinement without destabilizing response states already well represented by the backbone. The full-spectrum IR and Raman fidelity from small molecules to drug-like and biomolecular systems is preserved (see Figures 2–5 of the manuscript).

---

## Architecture

### EquiformerV2 Adaptation

The standard EquiformerV2 architecture is adapted to spectroscopic regression by removing the Open Catalyst Project dependency and replacing it with a lightweight radius-graph constructor tailored to the QM9S and QMe14S spatial scales. The Hessian blocks are computed by two derivative routes that share the same equivariant representation: a two-leaf mixed-derivative construction for H<sub>ii</sub> and an edge-wise derivative route for H<sub>ij</sub> that preserves the off-diagonal curvature signal. The polarizability branch uses the native rank-2 equivariant readout of EquiformerV2 rather than collapsing higher-order information into scalar invariants.

### ENK Bridge

ENK is a learned, equivariance-preserving confidence filter on the SO(3) latent representation. Filtering is applied only to the `l = 0` and `l = 2` readout channels (the linear readout paths); the quadratic Clebsch–Gordan self-coupling paths used by `l = 1` and `l ≥ 3` channels are left unmodified to avoid uncontrolled amplification. For each filtered degree, the state update, process uncertainty and observation-noise logit are predicted from the invariant features. When an atom-level electron prior is available, it modulates the observation-noise logit through a zero-initialized FiLM transformation. The Kalman gain is a scalar per atom per filtered degree, broadcast over magnetic components, thereby preserving rotational equivariance.

### Multi-Stage Electron Prior

The NBO predictor follows EMPP design principles: atomic geometry is encoded into atom features, expanded into a heterogeneous graph with atom, bond and lone-pair-related tokens, and refined by equivariant message passing and pair-aware aggregation. Training follows a three-stage transfer curriculum:

1. **Stage 1** — pretrain on SIMG (heterogeneous NBO graph supervision).
2. **Stage 2** — jointly optimize SIMG and qcMol to retain transferable electronic features while aligning to quantum-chemical observables.
3. **Stage 3** — shift to a qcMol-dominant regime with a small amount of SIMG replay to reduce forgetting.

Default atom targets: **NAO, LP, NPA** (with **ADCH, LI, ELF** as auxiliaries). Default bond target: **NBO** (with **DI, LBO, Mayer** as auxiliaries).

At inference time, the NBO predictor operates as a geometry-driven online electronic estimator. Its predictions are denormalized and mapped through learned adapters into two complementary prior fields: the **atom prior** π<sub>atom</sub> (summarizing local population and valence-state cues) and the **edge prior** π<sub>edge</sub> (encoding pairwise chemical information). These priors are injected at the EquiformerV2 transformer block (Point A) via the `SO3ElectronPriorInjector`.

### Guided Spectral Calibration

GSC is a bounded, branch-specific post-hoc calibration that acts on chemically localized residuals. Calibration statistics are computed offline from the training set (`compute_nbo_train_stats.py`) and applied at inference time per eligible response branch. The calibration is intentionally bounded so that it cannot override a well-supported backbone prediction; it only adjusts response states where the NBO prior provides a transferable correction direction.

---

## Repository Structure

```
SENK_main_4.10/
├── train_senk.py                  # Main training script (V2 + ENK + EP, all modes)
├── senk_train_shared.py           # Shared training utilities and dataset constants
├── dataloader_spectra.py          # Spectra dataloaders (QM9S / QMe14S, polar/dipole/vib)
├── prep_prior_cache.py            # Offline electron-prior cache precompute
├── compute_nbo_train_stats.py     # NBO-GSC training-statistics generator (Welford)
├── nbo_consistency_loss.py        # NBO-CR: Badger, acoustic sum rule, delocal-α, Born-α
├── nbo_spectral_calibration.py    # NBO-GSC bounded post-hoc calibrator (v1 / v2)
├── ep_canonical_defaults.py       # Canonical EP checkpoint / stats / default paths
├── v2_spectra_infer.py            # V2 spectra inference and plotting
├── optim_factory.py               # Optimizer factory (modified from timm)
├── logger.py                      # Lightweight file logger with rank0 filtering
├── requirements.txt               # Pinned dependency list
│
├── nets/                          # SENK core network modules
│   ├── clean_equiformer_polar.py          # V1 baseline: Equiformer + polarizability head
│   ├── clean_equiformer_polar_ext.py      # V2-native path: EP + SO3ENKBridge + V2SO3PolarReadout
│   ├── clean_equiformer_multitask.py      # Multi-task variant (polar / dipole / Hessian)
│   ├── equiformer_v2_backbone.py          # EquiformerV2 backbone (SO(2)-conv + S² activation)
│   ├── equivariant_neural_kalman.py       # ENK, TemporalENK, SO3ENKBridge (l=0, l=2 only)
│   ├── v2_electron_prior.py               # SO3ElectronPriorInjector (Point-A injection)
│   ├── v2_so3_polar_readout.py            # V2SO3PolarReadout (all L channels)
│   ├── tensor_aware_sdm.py                # Tensor-aware spectral mask attention
│   ├── graph_attention_transformer.py     # V1 Equiformer backbone
│   └── eqv2_core/                         # Vendored EquiformerV2 primitives (Facebook, MIT)
│       ├── so3.py                         # SO3_Embedding
│       ├── so2_ops.py                     # SO(2) convolutions
│       ├── transformer_block.py           # SO2EquivariantGraphAttention, TransBlockV2
│       ├── layer_norm.py                  # EquivariantLayerNormArray
│       ├── activation.py                  # ScaledSiLU, GateActivation, S2Activation
│       └── ...
│
├── nbo_nets/                      # NBO electron-prior predictor
│   ├── model_nbo_v2.py                    # NBOFoundationModel (EMPP design)
│   ├── nbo_loader.py                      # SIMG data loader
│   ├── nbo_modules.py                     # NBO model components
│   ├── nbo_shared_modules.py              # Shared equivariant token MP, distance encoder
│   ├── train_nbo_v2.py                    # Stage-1 SIMG pretraining
│   ├── train_nbo_joint.py                 # Stage-2/3 SIMG+qcMol joint training
│   ├── checkpoints/                       # Trained NBO predictor checkpoints
│   └── qcmol/                             # qcMol dataset and loader
│       ├── qcmol_loader.py                # qcMol data loader
│       ├── build_qcmol_fs_index.py        # qcMol filesystem index builder
│       └── preprocess_qcmol_packed_safe.py # qcMol preprocessing
│
├── detanet_nets/                  # DetaNet baseline (vendored, Nat. Comput. Sci. 2023)
│   ├── detanet.py                         # DetaNet model
│   ├── electron_prior.py                  # NBOPriorBranch (DetaNet-compatible EP)
│   ├── spectra_simulator.py               # IR / Raman / NMR spectral simulation
│   ├── model_loader.py                    # Model constructors (polar / dipole / Hi / Hij)
│   ├── constant.py                        # Physical constants and unit conversions
│   ├── metrics.py                         # Loss functions and metrics
│   ├── qm9spectra/                        # Pretrained DetaNet weights (QM9S)
│   └── modules/                           # DetaNet building blocks
│
├── datasets/                      # Dataset loaders and preprocessors
│   ├── qm9.py                             # QM9 loader (PyG-style, vendored)
│   └── QMe14S/
│       ├── qme14s_opt186102_extract.py    # QMe14S OPT_186102 raw extraction
│       ├── qme14s_opt186102_preprocess.py # QMe14S OPT_186102 preprocessing
│       └── README.txt                     # QMe14S dataset notes
│
└── tools/                         # Auxiliary tools
    ├── build_qm9s_skeletons.py            # Generate SENK skeleton files from qm9s.pt
    ├── build_skeleton_from_smiles.py      # Build skeleton from SMILES / SDF (RDKit ETKDGv3)
    ├── lp_predictor.py                    # Minimal SIMG lone-pair predictor (inference only)
    └── lp_pred_model.ckpt                 # Lone-pair predictor checkpoint
```

---

## Installation

SENK has been tested with Python 3.10, PyTorch 2.2.2 (CUDA 12.1) on Linux. We recommend using `conda` to manage the environment.

### Step 1 — Create the conda environment

```bash
conda create -n senk python=3.10 -y
conda activate senk
```

### Step 2 — Install PyTorch (cu121 wheels are forward-compatible with CUDA 12.2)

```bash
pip install torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — Install PyTorch Geometric C++ extensions (versions must match torch + CUDA exactly)

```bash
pip install torch-scatter torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

### Step 4 — Install PyTorch Geometric

```bash
pip install torch-geometric==2.5.3
```

### Step 5 — Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Verify the installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_geometric, torch_scatter, e3nn; print('PyG+e3nn OK')"
python -c "from nets.clean_equiformer_polar_ext import CleanEquiformerPolarExt; print('SENK OK')"
```

---

## Datasets

SENK uses four data sources. Each is described below together with the corresponding loader entry point.

### QM9S — reference small-molecule spectroscopy

QM9S provides direct supervision for the Hessian, dipole-derivative and polarizability-derivative heads. It is loaded via `dataloader_spectra.build_polar_dipole_loaders` and `dataloader_spectra.build_vib_loaders`.

### QMe14S — 14-element functional-group-rich chemical space

QMe14S is the main target dataset. The `datasets/QMe14S/` directory contains extraction and preprocessing scripts for the OPT_186102 trajectory split. After preprocessing (see [Data Preprocessing](#data-preprocessing)), QMe14S is consumed through the same loaders as QM9S.

### SIMG — heterogeneous NBO graph supervision

SIMG provides transferable electronic pretraining for the NBO predictor. It is loaded via `nbo_nets/nbo_loader.py`.

### qcMol — atom- and bond-resolved quantum-chemical annotations

qcMol refines the NBO predictor with quantum-chemical observables. The `nbo_nets/qcmol/` directory contains the loader, filesystem index builder and preprocessing tools. The training corpus combines a PubChemQC/ZINC-derived molecular subset with a PDBbind 2020 ligand subset.

> **Note**: Due to license considerations, raw QM9S / QMe14S / SIMG / qcMol data files are not bundled in this repository. Users should obtain them from the original sources cited in the manuscript and run the preprocessing scripts under `datasets/QMe14S/` and `nbo_nets/qcmol/` to generate the `.pt` shards expected by the loaders.

---

## Data Preprocessing

This section describes the sequential pipeline for converting raw data into training-ready `.pt` shards. Follow the steps in order. All commands assume the working directory is `SENK_main_4.10/`.

### Step 1 — QMe14S HDF5 extraction and preprocessing

The main QMe14S training dataset (`qme14s_opt186102.pt`) is generated from the raw `OPT_186102.h5` file in two stages: extraction followed by normalization.

**1a. Extract raw `.pt` list from HDF5**

```bash
python datasets/QMe14S/qme14s_opt186102_extract.py \
    --h5_path datasets/QMe14S/OPT_186102.h5 \
    --output datasets/QMe14S/qme14s_opt186102_raw.pt
```

> Quick test: add `--max_items 1000` to process only the first 1,000 entries.

**1b. Normalize to training format**

```bash
python datasets/QMe14S/qme14s_opt186102_preprocess.py \
    --input datasets/QMe14S/qme14s_opt186102_raw.pt \
    --output datasets/QMe14S/qme14s_opt186102.pt
```

> Optional: add `--max_atoms 30` to filter out molecules with more than 30 atoms (reduces GPU memory pressure).

**Output**: `datasets/QMe14S/qme14s_opt186102.pt` — directly consumable by the training loaders.

### Step 2 — Skeleton generation

SENK requires NBO skeleton files (atom/bond/lone-pair structural information) for both datasets. The same script `tools/build_qm9s_skeletons.py` handles both.

**2a. QM9S skeleton** (required when training on QM9S)

```bash
python tools/build_qm9s_skeletons.py \
    --qm9s datasets/qm9s.pt \
    --output_dir datasets/new_skeleton \
    --lp_ckpt tools/lp_pred_model.ckpt \
    --lp_device cuda
```

**Output**: `datasets/new_skeleton/skeleton_all.pt`

**2b. QMe14S skeleton** (required when training on QMe14S)

```bash
python tools/build_qm9s_skeletons.py \
    --qm9s datasets/QMe14S/qme14s_opt186102.pt \
    --output_dir datasets/QMe14S \
    --lp_ckpt tools/lp_pred_model.ckpt \
    --lp_device cuda
```

**Output**: `datasets/QMe14S/skeleton_all.pt`

### Step 3 — Electron prior cache precompute (EP modes only)

For training modes that use the electron prior (`equiformer_v2_enk_ep`, `equiformer_electron_prior`), pre-compute the NBO prior cache to avoid redundant forward passes during training:

```bash
python prep_prior_cache.py \
    --dataset qme14s_opt186102 \
    --task spectra4 \
    --electron-prior-mode simg \
    --electron-prior-ckpt nbo_nets/checkpoints/20260325-143400_joint_pdbbind_from_pubchem_joint/nbo_foundation_training_best.pt \
    --electron-prior-stats nbo_nets/checkpoints/20260325-143400_joint_pdbbind_from_pubchem_joint/norm_stats.pt \
    --shard-size 256 \
    --keep-shards 2 \
    --gpu 0
```

**Output**: `datasets/cache/electron_prior/electron_prior_cache_{hash}/` containing `meta.pt` and sharded `.pt` files for the train/val/test splits.

> Non-EP modes (`equiformer_v2`, `equiformer_v2_enk`) can skip this step.

### Step 4 — NBO-GSC training statistics (post-training, inference-only)

After training at least the H<sub>ij</sub>, ∂μ/∂R, and ∂α/∂R branches with ENK+EP, compute the per-bond/per-atom NBO feature statistics used by Guided Spectral Calibration at inference time:

```bash
python compute_nbo_train_stats.py \
    --dataset qme14s_opt186102 \
    --electron_prior_ckpt nbo_nets/checkpoints/20260325-143400_joint_pdbbind_from_pubchem_joint/nbo_foundation_training_best.pt \
    --electron_prior_stats nbo_nets/checkpoints/20260325-143400_joint_pdbbind_from_pubchem_joint/norm_stats.pt \
    --hij_ckpt checkpoints/equiformer_backbone/equiformer_v2_enk_ep/hij/{timestamp}/best.pt \
    --dd_ckpt checkpoints/equiformer_backbone/equiformer_v2_enk_ep/dedipole/{timestamp}/best.pt \
    --dp_ckpt checkpoints/equiformer_backbone/equiformer_v2_enk_ep/depolar/{timestamp}/best.pt \
    --output nbo_train_stats.pt \
    --gpu 0
```

**Output**: `nbo_train_stats.pt`

> This step logically belongs to data preparation but requires trained weights, so it runs after the training phase (see [Training](#training)).

### Pipeline summary

```
OPT_186102.h5
     │
     ├─ qme14s_opt186102_extract.py ────→ qme14s_opt186102_raw.pt
     │                                         │
     │                                         └─ qme14s_opt186102_preprocess.py ──→ qme14s_opt186102.pt
     │
     └─ tools/build_qm9s_skeletons.py ──→ skeleton_all.pt

prep_prior_cache.py ──────────────────→ datasets/cache/electron_prior/   (EP modes only)
compute_nbo_train_stats.py ───────────→ nbo_train_stats.pt               (post-training, GSC only)
```

---

## Pretrained Checkpoints

The following pretrained checkpoints are distributed with this repository.

| Checkpoint | Path | Purpose |
|---|---|---|
| NBO predictor (with links) | `nbo_nets/checkpoints/2025-11-10_11-06-07_nbo_model_with_links/nbo_model_with_links_best.pt` | Stage-3 NBO predictor used as the online electron prior. |
| NBO predictor norm stats | `nbo_nets/checkpoints/2025-11-10_11-06-07_nbo_model_with_links/nbo_model_with_links_norm_stats.pt` | Matching normalization statistics. |
| Lone-pair predictor | `tools/lp_pred_model.ckpt` | Inference-only SIMG lone-pair predictor used during skeleton construction. |
| DetaNet QM9S weights | `detanet_nets/qm9spectra/*.pth` | DetaNet baseline weights (H<sub>ii</sub>, H<sub>ij</sub>, dipole, polar, depolar, dedipole, etc.) for hybrid Raman inference. |

The main SENK response-state checkpoints (V2 + ENK + EP, trained on QM9S / QMe14S) are released alongside the manuscript. Please refer to the manuscript's "Data availability" statement for the download links.

---

## Training

SENK supports five training modes through `train_senk.py`:

| Mode | Backbone | ENK | EP | Description |
|---|---|---|---|---|
| `clean_equiformer` | V1 Equiformer | — | — | V1 baseline. |
| `equiformer_electron_prior` | V1 Equiformer | — | ✓ | V1 + EP (scalar attention). |
| `equiformer_v2` | EquiformerV2 | — | — | V2 backbone baseline. |
| `equiformer_v2_enk` | EquiformerV2 | ✓ | — | V2 + ENK (no EP). |
| `equiformer_v2_enk_ep` | EquiformerV2 | ✓ | ✓ | **Full SENK** (V2 + ENK + EP). |

### Example — train full SENK on QMe14S

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

The script auto-detects the backbone architecture from a checkpoint when resuming (`--resume`), and supports branch-specific training for `polar`, `depolar`, `sobolev_polar`, `dipole`, `dedipole`, `hii`, `hij` and `spectra4`. Sobolev joint training computes analytic Jacobian floors via autograd (`_jacobian_floor_loss`).

### Training the NBO electron prior

The three-stage curriculum is launched through two scripts:

```bash
# Stage 1 — SIMG pretraining
python nbo_nets/train_nbo_v2.py --config configs/nbo_simg.yaml

# Stage 2 / 3 — SIMG + qcMol joint training
python nbo_nets/train_nbo_joint.py --config configs/nbo_joint.yaml
```

### Computing NBO-GSC training statistics

Before running EP inference, the GSC calibration statistics must be precomputed once per training set:

```bash
python compute_nbo_train_stats.py \
    --hij_ckpt outputs/senk_v2_enk_ep_qme14s_hij/best.pt \
    --dd_ckpt  outputs/senk_v2_enk_ep_qme14s_dd/best.pt \
    --dp_ckpt  outputs/senk_v2_enk_ep_qme14s_dp/best.pt \
    --output   outputs/nbo_train_stats.pt
```

### Precomputing the electron-prior cache

For large datasets, the NBO predictor outputs can be cached offline to avoid recomputation during SENK training:

```bash
python prep_prior_cache.py \
    --dataset qme14s_opt186102 \
    --electron_prior_mode qcmol \
    --electron_prior_ckpt nbo_nets/checkpoints/2025-11-10_11-06-07_nbo_model_with_links/nbo_model_with_links_best.pt \
    --electron_prior_cache_dir datasets/cache/electron_prior
```

---

## Inference

### Single-molecule spectra inference (V2 + ENK + EP)

`v2_spectra_infer.py` is the EquiformerV2 self-trained four-weights spectra inference program. It loads the four SENK branch checkpoints (H<sub>ii</sub>, H<sub>ij</sub>, dedipole, depolar), runs a forward pass on a single molecule, and produces IR and Raman spectra together with diagnostic plots.

```bash
python v2_spectra_infer.py \
    --xyz examples/molecule.xyz \
    --hii_ckpt  outputs/senk_v2_enk_ep_qme14s_hii/best.pt \
    --hij_ckpt  outputs/senk_v2_enk_ep_qme14s_hij/best.pt \
    --dedipole_ckpt outputs/senk_v2_enk_ep_qme14s_dedipole/best.pt \
    --depolar_ckpt  outputs/senk_v2_enk_ep_qme14s_depolar/best.pt \
    --nbo_ckpt nbo_nets/checkpoints/2025-11-10_11-06-07_nbo_model_with_links/nbo_model_with_links_best.pt \
    --nbo_stats nbo_nets/checkpoints/2025-11-10_11-06-07_nbo_model_with_links/nbo_model_with_links_norm_stats.pt \
    --gsc_stats outputs/nbo_train_stats.pt \
    --out outputs/spectrum.png
```

### Building a skeleton from a novel molecule

For inference on molecules outside QM9S / QMe14S, a skeleton file must first be built from SMILES or SDF:

```bash
python tools/build_skeleton_from_smiles.py \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --out outputs/aspirin_skeleton.pt
```

The skeleton file contains the atom / bond / lone-pair token graph required by the NBO electron-prior predictor.

---

## Citation

If you find this repository useful for your research, please consider citing the accompanying manuscript:



This repository also builds on the following open-source projects, which we gratefully acknowledge:

- **EquiformerV2** — Liao et al., ICLR 2024. [https://github.com/atomicarchitects/equiformer_v2](https://github.com/atomicarchitects/equiformer_v2)
- **DetaNet** — Zou et al., *Nat. Comput. Sci.* 3, 957–964 (2023).
- **e3nn** — Geiger, Smidt et al. [https://e3nn.org](https://e3nn.org)
- **PyTorch Geometric** — Fey & Lenssen. [https://pyg.org](https://pyg.org)
- **RDKit** — Landrum et al. [https://www.rdkit.org](https://www.rdkit.org)

---

## Acknowledgements

We thank the developers of EquiformerV2, DetaNet, e3nn, PyTorch Geometric and RDKit for making their code publicly available. We also thank the contributors of the QM9S, QMe14S, SIMG and qcMol datasets. Funding sources and additional acknowledgements are listed in the manuscript.

---

## License

This project is released under the **MIT License**. The vendored components under `nets/eqv2_core/` (EquiformerV2, Copyright (c) Meta, Inc.) and `detanet_nets/` (DetaNet) retain their original upstream licenses, as indicated in each file header.
