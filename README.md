# Adversarial Robustness Framework for Vision Transformers

**UTA AI Course Project** — Benchmarking and improving the adversarial robustness of Vision Transformers (ViT) vs. ResNet-18 on CIFAR-10.

**Team:** Bhargav Amin · Allen Joshua Selvaraj · Somesh Zanwar

---

## Overview

This project investigates how robust Vision Transformers are against adversarial attacks compared to traditional CNNs. We implement three attack methods and one defense strategy, and provide an interactive Gradio demo to visualize adversarial examples in real time.

| Model | Architecture | Parameters |
|-------|-------------|------------|
| ViT | DeiT-Small (patch16, 224) | ~22M |
| CNN Baseline | ResNet-18 | ~11M |

---

## Results

| Model | Clean | FGSM | PGD | Patch |
|-------|-------|------|-----|-------|
| ViT (Standard) | 98.4% | 35.3% | 0.0% | 9.0% |
| ViT (Adv. Trained) | 90.8% | 4.4% | 0.0% | 0.4% |
| ResNet-18 (Standard) | 97.3% | 30.4% | 0.0% | 33.4% |
| ResNet-18 (Adv. Trained) | 18.7% | 8.0% | 2.5% | 10.8% |

---

## Project Structure

```
adversarial-robustness-vit/
├── configs/
│   └── config.yaml              # All hyperparameters
├── data/
│   └── loader.py                # CIFAR-10 dataloader (auto-downloads)
├── models/
│   ├── vit_model.py             # DeiT-Small wrapper (timm)
│   └── resnet_model.py          # ResNet-18 wrapper (torchvision)
├── attacks/
│   ├── fgsm.py                  # Fast Gradient Sign Method
│   ├── pgd.py                   # Projected Gradient Descent
│   └── patch_attack.py          # Adversarial Patch
├── defenses/
│   └── adversarial_training.py  # PGD adversarial training (Madry et al.)
├── evaluation/
│   └── metrics.py               # Clean + robust accuracy evaluation
├── demo/
│   └── app.py                   # Interactive Gradio demo
├── sample_images/               # One test image per CIFAR-10 class
├── checkpoints/                 # Saved model weights
├── results/                     # Evaluation results and plots
├── train.py                     # Fine-tuning + adversarial training
├── evaluate.py                  # Benchmarking + result plots
└── main.py                      # Single entry point for all commands
```

---

## Quick Start

See **[SETUP.md](SETUP.md)** for full step-by-step instructions (especially if you downloaded the zip).

### Install dependencies

**CPU:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**GPU (CUDA 12.4):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Run

```bash
python main.py train        # Fine-tune ViT + ResNet on CIFAR-10
python main.py evaluate     # Benchmark robustness (saves to results/)
python main.py demo         # Launch Gradio demo at http://localhost:7860
```

---

## Attacks

| Attack | Type | Parameters |
|--------|------|------------|
| FGSM | Single-step gradient | ε = 0.03 |
| PGD | Iterative multi-step | ε = 0.03, α = 0.007, steps = 40 |
| Adversarial Patch | Localized patch | 32×32 patch, 100 iterations |

## Defense

**PGD Adversarial Training (Madry et al., 2018)** — trains on 7-step PGD adversarial examples generated on-the-fly each mini-batch.

---

## Hardware

Tested on:
- CPU: Intel Core i7 12th Gen
- GPU: NVIDIA RTX 3050 Ti Laptop (4 GB VRAM)
- Python 3.10+, PyTorch 2.2, CUDA 12.4

Training time on GPU: ~10–20 min per model.

---

## References

- Madry et al. (2018) — *Towards Deep Learning Models Resistant to Adversarial Attacks*
- Dosovitskiy et al. (2020) — *An Image is Worth 16x16 Words*
- Touvron et al. (2021) — *Training data-efficient image transformers (DeiT)*
- Goodfellow et al. (2015) — *Explaining and Harnessing Adversarial Examples*
- torchattacks — https://github.com/Harry24k/adversarial-attacks-pytorch
