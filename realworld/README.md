# Real-World Adversarial Robustness Module

This is a **standalone extension** of the main project that works on **any real-world photo** — humans, animals, vehicles, objects — using ImageNet-pretrained models.

> **To remove this module entirely:** just delete the `realworld/` folder. Nothing else in the project depends on it.

---

## What It Does

| Feature | Details |
|---------|---------|
| Models | DeiT-Small (ViT) + ResNet-18, pretrained on ImageNet-1k |
| Classes | 1000 real-world categories |
| Auto-labelling | Predicts category: **Human / Animal / Plant / Non-Living Object** |
| Attacks | FGSM, PGD, Adversarial Patch |
| Training required | **No** — uses pretrained weights out of the box |

---

## Setup

Follow the main project setup first (install dependencies, activate `.venv`).

> Make sure you're in the **project root** (`adversarial-robustness-vit/`), not inside `realworld/`.

```bash
# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

---

## Running the Demo

```bash
python realworld/demo.py
```

Then open **http://localhost:7860** in your browser.

---

## How to Use the Demo

1. **Upload any real photo** — a person, a dog, a car, a landscape, anything
2. **Select a model** — ViT (DeiT-Small) or ResNet-18
3. **Select an attack** — None / FGSM / PGD / Patch
4. **Adjust Epsilon** — controls how strong the perturbation is (lower = less visible)
5. Click **Analyse**

### Output
| Panel | Description |
|-------|-------------|
| Original Image | Your uploaded photo |
| Adversarial Image | Perturbed version (looks nearly identical to human eye) |
| Clean Prediction | What the model sees before attack |
| Adversarial Prediction | What the model sees after attack |
| Top-K Table | Full ranked predictions with confidence + category label |

---

## File Structure

```
realworld/
├── loader.py       # Image preprocessing (ImageNet normalization, any resolution)
├── models.py       # Load pretrained ViT + ResNet (no training needed)
├── classify.py     # Top-k predictions + Human/Animal/Plant/Object labelling
├── demo.py         # Gradio web demo
└── README.md       # This file
```

---

## Example Results

| Image | Clean Prediction | After FGSM (ε=0.02) |
|-------|-----------------|---------------------|
| Photo of a dog | golden retriever (92%) [Animal] | Persian cat (61%) [Animal] |
| Photo of a car | sports car (88%) [Non-Living] | racer (74%) [Non-Living] |
| Photo of a person | suit (76%) [Human] | lab coat (58%) [Human] |

---

## Notes

- **Epsilon (ε):** Keep between `0.01–0.05` for realistic attacks (perturbation invisible to humans)
- **Patch attack** ignores epsilon — it optimises a visible patch in a corner of the image
- The category labelling uses keyword matching against all 1000 ImageNet class names
- For best results, use clear, well-lit photos at reasonable resolution (≥ 224×224)
