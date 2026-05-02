# Setup & Run Guide

Step-by-step instructions for running the project after downloading the zip.

---

## Requirements

- Python 3.10 or higher
- pip
- (Optional but recommended) NVIDIA GPU with CUDA 12.x for faster training

---

## Step 1 — Extract the zip

Extract the downloaded zip file to any folder on your machine. You should see a folder structure like:

```
adversarial-robustness-vit/
├── configs/
├── data/
├── models/
├── attacks/
├── defenses/
├── evaluation/
├── demo/
├── checkpoints/
├── results/
├── sample_images/
├── main.py
├── train.py
├── evaluate.py
└── requirements.txt
```

Open a terminal (Command Prompt or PowerShell on Windows) and navigate into the folder:

```bash
cd path\to\adversarial-robustness-vit
```

---

## Step 2 — Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / Mac
source .venv/bin/activate
```

You should see `(.venv)` appear at the start of your terminal prompt.

---

## Step 3 — Install PyTorch

**If you have a NVIDIA GPU (recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CPU only (slower, but works):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5 — Train the models

This downloads CIFAR-10 automatically (~170 MB) and fine-tunes both ViT and ResNet-18.

```bash
python main.py train
```

This takes about 10–20 minutes on GPU or 1–2 hours on CPU. Checkpoints are saved to `checkpoints/`.

> **Skip this step** if `checkpoints/vit_cifar10.pth` and `checkpoints/resnet_cifar10.pth` already exist in the zip (pre-trained weights included).

**Optional — run adversarial training on top:**
```bash
python main.py train --adv-train
```

---

## Step 6 — Evaluate robustness

```bash
python main.py evaluate
```

This runs clean accuracy + FGSM + PGD + Patch attack evaluation on 1,000 test images. Results are saved to `results/results.json` and a bar chart to `results/robustness_comparison.png`.

> **Skip this step** if `results/results.json` already exists — pre-computed results are included.

---

## Step 7 — Launch the interactive demo

```bash
python main.py demo
```

Open **http://localhost:7860** in your browser. You can:

1. Upload any CIFAR-10 style image (or use one from `sample_images/`)
2. Select a model — ViT or ResNet-18
3. Select an attack — None / FGSM / PGD / Patch
4. Adjust the epsilon slider (perturbation strength)
5. Click **Run** to see the adversarial image and how the prediction changes

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Make sure `.venv` is activated and `pip install -r requirements.txt` was run |
| `CUDA out of memory` | Reduce `batch_size` in `configs/config.yaml` (try 32 or 16) |
| CIFAR-10 download fails | Check internet connection; data saves to `data/datasets/` |
| PowerShell execution policy error | Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Port 7860 already in use | Stop any other Gradio apps or change the port in `demo/app.py` |

---

## File Overview

| File / Folder | Purpose |
|---------------|---------|
| `main.py` | Entry point — runs train / evaluate / demo |
| `train.py` | Fine-tuning and adversarial training logic |
| `evaluate.py` | Evaluation pipeline, saves results and plots |
| `configs/config.yaml` | All hyperparameters (epochs, epsilon, batch size, etc.) |
| `checkpoints/` | Saved model weights after training |
| `results/` | JSON results table + robustness comparison chart |
| `sample_images/` | 10 sample images (one per CIFAR-10 class) for the demo |
