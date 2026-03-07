# Adversarial Robustness Framework for Vision Transformers

**UTA AI Course Project** — A framework to benchmark and improve the adversarial robustness of Vision Transformers (ViT) vs CNN baselines on CIFAR-10.

---

## Overview

This project investigates how robust Vision Transformers are against adversarial attacks compared to traditional CNNs (ResNet-18). We implement three attack methods, one defense strategy, and provide an interactive demo to visualize adversarial examples in real time.

| Model | Architecture | Params |
|-------|-------------|--------|
| ViT   | DeiT-Small (patch16, 224) | ~22M |
| CNN Baseline | ResNet-18 | ~11M |

---

## Project Structure

```
adversarial-robustness-vit/
├── configs/
│   └── config.yaml              # All hyperparameters in one place
├── data/
│   └── loader.py                # CIFAR-10 dataloader (auto-downloads)
├── models/
│   ├── vit_model.py             # DeiT-Small wrapper (timm)
│   └── resnet_model.py          # ResNet-18 wrapper (torchvision)
├── attacks/
│   ├── fgsm.py                  # Fast Gradient Sign Method
│   ├── pgd.py                   # Projected Gradient Descent
│   └── patch_attack.py          # Adversarial Patch (custom)
├── defenses/
│   └── adversarial_training.py  # PGD adversarial training (Madry et al.)
├── evaluation/
│   └── metrics.py               # Clean + robust accuracy evaluation
├── demo/
│   └── app.py                   # Interactive Gradio web demo
├── sample_images/               # One test image per CIFAR-10 class
├── train.py                     # Fine-tuning + adversarial training
├── evaluate.py                  # Benchmarking + result plots
└── main.py                      # Single entry point for all commands
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/bhargav0504/adversarial-robustness-vit.git
cd adversarial-robustness-vit
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**GPU (CUDA 12.4 — recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Fine-tune models on CIFAR-10
```bash
python main.py train
```
Trains both DeiT-Small and ResNet-18 on CIFAR-10. Saves checkpoints to `checkpoints/`.

### Step 2 — Adversarial training (optional but recommended)
```bash
python main.py train --adv-train
```
Runs PGD-based adversarial training after standard fine-tuning.

### Step 3 — Evaluate robustness
```bash
python main.py evaluate
```
Benchmarks clean accuracy + robustness under FGSM, PGD, and Patch attacks.
Saves a comparison plot and JSON table to `results/`.

### Step 4 — Interactive demo
```bash
python main.py demo
```
Opens a Gradio web app at `http://localhost:7860`. Upload any image, pick an attack, and see the adversarial example and perturbed prediction live.

---

## Attacks Implemented

| Attack | Type | Key Parameter |
|--------|------|---------------|
| **FGSM** | Single-step gradient | ε (epsilon) |
| **PGD** | Multi-step iterative | ε, α (step size), steps |
| **Adversarial Patch** | Localized patch | patch size, iterations |

---

## Defense

**Adversarial Training (Madry et al., 2018)** — The model is trained on PGD-generated adversarial examples instead of clean images. This is the standard baseline for certified robustness.

---

## Hardware

Tested on:
- CPU: Intel Core i7 12th Gen
- GPU: NVIDIA RTX 3050 Ti Laptop (4GB VRAM)
- Python 3.13, PyTorch 2.6, CUDA 12.4

Training time on GPU: ~10–20 min per model on CIFAR-10.

---

## Team

UTA AI Course Project — Group of 3

---

## References

- Madry et al. (2018) — *Towards Deep Learning Models Resistant to Adversarial Attacks*
- Dosovitskiy et al. (2020) — *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
- Touvron et al. (2021) — *Training data-efficient image transformers & distillation through attention (DeiT)*
- `torchattacks` library — https://github.com/Harry24k/adversarial-attacks-pytorch
