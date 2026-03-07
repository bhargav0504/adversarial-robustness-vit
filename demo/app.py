"""
demo/app.py — Interactive Gradio demo for the Adversarial Robustness Framework.

Run from the project root:
  python demo/app.py

Then open http://localhost:7860 in your browser.
"""
import os
import sys

import numpy as np
import torch
import torchattacks
import yaml
from PIL import Image
from torchvision import transforms

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from attacks.patch_attack import PatchAttack
from data.loader import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from models import load_resnet, load_vit

# ── Config & device ──────────────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')) as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Preprocessing ─────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def tensor_to_pil(tensor):
    """Convert a normalised (C,H,W) or (1,C,H,W) tensor to PIL."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img  = (tensor.squeeze(0).cpu() * std + mean).clamp(0, 1)
    return Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def load_model(model_choice):
    cfg = CONFIG.copy()
    if model_choice == 'ViT (DeiT-Small)':
        ckpt = './checkpoints/vit_cifar10.pth'
        cfg['models']['vit']['checkpoint'] = ckpt if os.path.exists(ckpt) else None
        return load_vit(cfg, DEVICE)
    else:
        ckpt = './checkpoints/resnet_cifar10.pth'
        cfg['models']['resnet']['checkpoint'] = ckpt if os.path.exists(ckpt) else None
        return load_resnet(cfg, DEVICE)


# ── Main inference function ───────────────────────────────────────────────────
def run_attack(pil_image, model_choice, attack_choice, eps):
    if pil_image is None:
        return None, None, "No image provided", "No image provided", "—"

    model = load_model(model_choice)
    model.eval()

    img_t = TRANSFORM(pil_image.convert('RGB')).unsqueeze(0).to(DEVICE)

    # ── Clean prediction ──────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(img_t)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    clean_label = f"{CIFAR10_CLASSES[pred]}  ({probs[pred]*100:.1f}%)"
    target      = torch.tensor([pred]).to(DEVICE)

    # ── Generate adversarial example ──────────────────────────────────────────
    if attack_choice == 'FGSM':
        atk = torchattacks.FGSM(model, eps=eps)
        adv_t = atk(img_t, target)
    elif attack_choice == 'PGD':
        atk = torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=40)
        adv_t = atk(img_t, target)
    else:  # Patch
        atk   = PatchAttack(model, patch_size=32, max_iter=50, device=DEVICE)
        adv_t = atk(img_t, target)

    adv_pil = tensor_to_pil(adv_t)

    # ── Adversarial prediction ────────────────────────────────────────────────
    with torch.no_grad():
        adv_logits = model(adv_t)
        adv_probs  = torch.softmax(adv_logits, dim=1)[0]
        adv_pred   = adv_probs.argmax().item()

    adv_label = f"{CIFAR10_CLASSES[adv_pred]}  ({adv_probs[adv_pred]*100:.1f}%)"

    # ── Perturbation visualisation (amplified 10×) ────────────────────────────
    pert     = (adv_t - img_t).abs()
    pert_pil = tensor_to_pil((pert * 10).clamp(0, 1))

    status = "FOOLED ✗" if pred != adv_pred else "Robust ✓  (prediction unchanged)"

    return adv_pil, pert_pil, clean_label, adv_label, status


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Adversarial Robustness Framework") as demo:
    gr.Markdown(
        "# Adversarial Robustness Framework\n"
        "**UTA AI Course Project** — Upload any CIFAR-10 style image and see "
        "how adversarial attacks change the model's prediction.\n\n"
        "> Note: Run `python train.py` first to fine-tune the models on CIFAR-10. "
        "Without checkpoints, predictions will be random (untrained head)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp_image    = gr.Image(type='pil', label="Input Image")
            model_choice = gr.Radio(
                ['ViT (DeiT-Small)', 'ResNet-18'],
                value='ViT (DeiT-Small)', label="Model"
            )
            attack_choice = gr.Radio(
                ['FGSM', 'PGD', 'Patch'],
                value='FGSM', label="Attack"
            )
            eps_slider = gr.Slider(
                0.01, 0.3, value=0.03, step=0.01,
                label="Epsilon  (perturbation strength, ignored for Patch)"
            )
            run_btn = gr.Button("Run Attack", variant="primary")

        with gr.Column(scale=1):
            out_adv   = gr.Image(label="Adversarial Image")
            out_pert  = gr.Image(label="Perturbation  (amplified 10×)")
            clean_out = gr.Textbox(label="Clean Prediction")
            adv_out   = gr.Textbox(label="Adversarial Prediction")
            status_out = gr.Textbox(label="Result")

    run_btn.click(
        run_attack,
        inputs=[inp_image, model_choice, attack_choice, eps_slider],
        outputs=[out_adv, out_pert, clean_out, adv_out, status_out],
    )

if __name__ == '__main__':
    demo.launch(share=False)
