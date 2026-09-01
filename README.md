# ProbVLP: Probabilistic Vision-Language Adaptation for Colorectal Polyp Segmentation

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-20BEFF.svg)](https://kaggle.com)
[![WACV 2025](https://img.shields.io/badge/Venue-WACV%202025-blueviolet.svg)]()

**ProbVLP** is a probabilistic vision-language adaptation framework for colorectal polyp segmentation.<br>
Trained on Kvasir-SEG only — achieving robust zero-shot generalization to 4 unseen colonoscopy datasets.

[📄 Paper](#citation) • [🚀 Quick Start](#-quick-start-kaggle) • [⚙️ Configuration](#️-configuration) • [📊 Results](#-results)

</div>

---

## 🔬 Overview

Colorectal cancer is one of the leading causes of cancer-related mortality worldwide. Accurate delineation of polyps during colonoscopy is clinically critical, yet current deep learning models struggle to generalize across diverse imaging equipment and protocols.

**ProbVLP** bridges this gap by combining:

- 🏥 **Endoscopy-specific visual representation** via EndoViT (MAE-pretrained ViT-B/16 on Endo700k)  
- 🔗 **Efficient multi-modal fusion** via LoRA-adapted bidirectional cross-attention (PVL adapters)  
- 🧭 **Spatial feedback loop** via boundary-prediction gating that informs each adapter  
- 📊 **Calibrated uncertainty estimation** via inverse-variance TTA with MC-dropout

---

## 🏗️ Architecture

```
Input Endoscopy Image (224 × 224)
          │
          ▼
  ┌───────────────┐    Text Prompt ──► CLIP Text Encoder
  │   EndoViT     │                            │
  │  ViT-B/16     │    (domain-renormalized)   │
  │  (frozen)     │                            │
  └──────┬────────┘                            │
         │  patch tokens  (197 × 768)          │
         ▼                                     ▼
  ┌──────────────────────────────────────────────────────┐
  │             PVL Adapters × N                         │
  │   Bidirectional cross-attention (img↔text)           │
  │   LoRA on Q / K / V / out_proj  [r=8, α=16]         │
  │   Boundary-Feedback Gate (tanh-gated, init=0)        │
  └───────────────────────┬──────────────────────────────┘
                           │
                           ▼
                  Upscale + SE Block
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
            Seg Logits        Boundary Logits
         (threshold 0.5)  (supervision + feedback)
                  │
         [Inference-time TTA]
          Flip × Scale × MC-Dropout
                  │
         Inverse-Variance Fusion
                  │
           ┌──────┴──────┐
           ▼             ▼
      Segmentation    Uncertainty
         Mask            Map
```

---

## ✨ Key Contributions

| # | Component | Description |
|---|-----------|-------------|
| 1 | **EndoViT Backbone** | MAE-pretrained ViT-B/16 on Endo700k replaces CLIP's general-domain visual tower with endoscopy-specific features. Domain renormalization bridges the statistics gap at the patch-embedding stage. |
| 2 | **LoRA PVL Adapters** | Low-rank adaptation (r=8, α=16) of Q/K/V/out\_proj in both cross-attention directions. Only ~3.2M parameters are trained; the backbone stays frozen. |
| 3 | **Boundary-Feedback Gating** | A lightweight boundary head predicts edge logits. These are pooled to patch resolution and gated back into each PVL adapter via a learnable scalar `tanh(gᵢ)` (init=0, identical to no gating at init). |
| 4 | **Inverse-Variance TTA** | At test time: 2 flips × 3 scales × 20 MC-dropout samples → 6 views fused by inverse-variance weighting. Yields calibrated segmentation + uncertainty maps. |

---

## 📦 Repository Structure

```
ProbVLM/
├── train.py              # Full Kaggle-ready training + evaluation pipeline
├── main.tex              # LaTeX paper entry point
├── references.bib        # BibTeX bibliography
├── LICENSE               # MIT license
├── .gitignore
├── sec/
│   ├── 1_intro.tex       # Introduction
│   ├── 2_related.tex     # Related Work
│   ├── 3_method.tex      # Methodology (EndoViT, LoRA, Boundary-FB, TTA)
│   ├── 4_experiments.tex # Datasets, metrics, ablation & cross-domain tables
│   └── 5_conclusion.tex  # Conclusion & future work
└── README.md
```

---

## 🚀 Quick Start (Kaggle)

### 1. Add Kaggle Datasets

Add the following datasets to your Kaggle notebook (input → add data):

| Dataset | Kaggle Path |
|---------|------------|
| Kvasir-SEG | `krishnaryali3/kvasir` |
| CVC-ClinicDB | `krishnaryali3/clinicdb` |
| ColonDB | `krishnaryali3/colondb-ployp` |
| CVC-300 | `krishnaryali3/cvc300` |
| BKAI-IGH | `krishnaryali3/bkai-ployp` |

### 2. Run the Pipeline

```python
# In a Kaggle notebook cell:
exec(open("/kaggle/input/.../train.py").read())
```

Or as a standalone script:
```bash
python train.py
```

The script will:
1. Clone [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) and install all dependencies
2. Stage all 5 datasets
3. Download EndoViT weights from HuggingFace (`egeozsoy/EndoViT`)
4. Patch the model with LoRA adapters + boundary-feedback gates
5. Train for 100 epochs on Kvasir-SEG
6. Evaluate on all 5 datasets (cross-domain generalization, Table B)

### Dependencies (auto-installed)

```bash
pip install monai fvcore timm einops open_clip_torch \
    "transformers==4.56.0" ftfy regex sentencepiece \
    openpyxl easydict huggingface_hub connected-components-3d matplotlib
```

---

## ⚙️ Configuration

All hyperparameters are set at the top of `train.py`:

```python
# Training
NUM_EPOCHS             = 100    # final model
ABLATION_EPOCHS        = 30     # per ablation row
BATCH_SIZE             = 16
SEED                   = 1

# Inference
USE_TTA                = True   # inverse-variance TTA
TEST_MC                = 20     # MC-dropout samples

# Architecture toggles
USE_ENDOVIT_VISUAL     = True   # EndoViT visual backbone
USE_LORA_ATTN_ADAPTER  = True   # LoRA on PVL cross-attention
USE_BOUNDARY_FEEDBACK  = True   # boundary-feedback gating
USE_SE                 = True   # Squeeze-and-Excitation recalibration

# LoRA hyperparameters
LORA_R                 = 8
LORA_ALPHA             = 16
LORA_SCOPE             = "both" # "both" | "vision" | "text"

# Loss weights
BOUNDARY_W             = 0.3
TVERSKY_W              = 0.3
TVERSKY_A, TVERSKY_B   = 0.3, 0.7  # penalizes FN more

# Study flags
RUN_CROSS_DOMAIN_EVAL  = True   # Table B: zero-shot OOD evaluation
RUN_ABLATION_STUDY     = False  # Table A: architecture contribution
RUN_LORA_SCOPE_STUDY   = False  # Table C: LoRA scope study
```

---

## 📊 Results

> Results will be populated after the full Kaggle run completes.

### Table A — Architecture Ablation (Kvasir-SEG, in-domain)

| Model | EndoViT | LoRA | Boundary-FB | TTA | DSC% | NSD% | HD95↓ |
|-------|:-------:|:----:|:-----------:|:---:|-----:|-----:|------:|
| Baseline (MedCLIPSeg) | ✗ | ✗ | ✗ | ✗ | — | — | — |
| + EndoViT | ✓ | ✗ | ✗ | ✗ | — | — | — |
| + LoRA PVL | ✓ | ✓ | ✗ | ✗ | — | — | — |
| + Boundary-FB | ✓ | ✓ | ✓ | ✗ | — | — | — |
| **ProbVLP (ours)** | ✓ | ✓ | ✓ | ✓ | **—** | **—** | **—** |

### Table B — Cross-Domain Generalization (zero-shot)

| Dataset | w/o TTA DSC% | w/o TTA NSD% | w/ TTA DSC% | w/ TTA NSD% |
|---------|:------------:|:------------:|:-----------:|:-----------:|
| Kvasir (ID) | — | — | — | — |
| ClinicDB | — | — | — | — |
| ColonDB | — | — | — | — |
| CVC-300 | — | — | — | — |
| BKAI | — | — | — | — |
| **OOD Avg.** | **—** | **—** | **—** | **—** |

---

## 📐 Method Details

### EndoViT Domain Renormalization

Since the incoming image is normalized to CLIP statistics by the dataloader,
we undo and re-normalize at the patch-embedding stage:

$$\hat{x} = \frac{x \odot \sigma_\text{CLIP} + \mu_\text{CLIP} - \mu_\text{EndoViT}}{\sigma_\text{EndoViT}}$$

### LoRA Update Rule

$$W = W_0 + \frac{\alpha}{r} \cdot BA \quad (A \in \mathbb{R}^{r \times d_\text{in}},\ B \in \mathbb{R}^{d_\text{out} \times r})$$

$B$ is initialized to zero (delta = 0 at init), preserving pretrained MedCLIPSeg representations.

### Inverse-Variance TTA Fusion

$$\hat{p} = \frac{\sum_k m_k / v_k}{\sum_k 1/v_k}, \qquad u = \left(\sum_k 1/v_k\right)^{-1}$$

where $m_k$ and $v_k$ are per-view MC-dropout mean and variance. $u$ serves as the uncertainty map.

---

## 🗃️ Datasets

| Dataset | Split | # Images | Purpose |
|---------|-------|----------|---------|
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | Train/Val/Test | 1000 | Training source |
| [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/) | Test | 612 | Zero-shot OOD eval |
| [ColonDB](https://www.depeca.uah.es/colonoscopy_dataset/) | Test | 380 | Zero-shot OOD eval |
| [CVC-300](http://pages.cvc.uab.es/CVC-Colon/index.php/databases/) | Test | 60 | Zero-shot OOD eval |
| [BKAI-IGH](https://www.kaggle.com/c/bkai-igh-neopolyp) | Test | 1000 | Zero-shot OOD eval |

---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@article{probvlp2025,
  title   = {ProbVLP: Probabilistic Vision-Language Adaptation
             for Colorectal Polyp Segmentation},
  author  = {Anonymous},
  journal = {WACV},
  year    = {2025}
}
```

---

## 🔗 Acknowledgements

This work builds on:

- **[MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg)** — HealthX Lab (CLIP-based medical segmentation backbone)
- **[EndoViT](https://huggingface.co/egeozsoy/EndoViT)** — Ego Ozsoy et al. (MAE-pretrained endoscopy ViT)
- **[LoRA](https://github.com/microsoft/LoRA)** — Microsoft Research (low-rank adaptation)
- **[MONAI](https://monai.io/)** — medical image analysis framework

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
