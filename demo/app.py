import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchattacks
import yaml
from PIL import Image
from torchvision import transforms

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GRADIO_TMP   = os.path.join(_PROJECT_ROOT, '.gradio_tmp')
os.makedirs(_GRADIO_TMP, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = _GRADIO_TMP

sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr

from attacks.patch_attack import PatchAttack
from data.loader import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from models import load_resnet, load_vit

with open(os.path.join(_PROJECT_ROOT, 'configs', 'config.yaml')) as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Attacks need [0, 1] inputs so we don't normalize here
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_MEAN = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
_STD  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)


class NormalizedModel(nn.Module):
    """Wraps model with normalization so attacks can work in [0, 1] space."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', _MEAN.clone())
        self.register_buffer('std',  _STD.clone())

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def tensor_to_pil(tensor):
    img = tensor.squeeze(0).cpu().clamp(0, 1)
    return Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def load_model(model_choice):
    cfg = CONFIG.copy()
    if model_choice == 'ViT (DeiT-Small)':
        ckpt = os.path.join(_PROJECT_ROOT, 'checkpoints', 'vit_cifar10.pth')
        cfg['models']['vit']['checkpoint'] = ckpt if os.path.exists(ckpt) else None
        model = load_vit(cfg, DEVICE)
    else:
        ckpt = os.path.join(_PROJECT_ROOT, 'checkpoints', 'resnet_cifar10.pth')
        cfg['models']['resnet']['checkpoint'] = ckpt if os.path.exists(ckpt) else None
        model = load_resnet(cfg, DEVICE)
    return NormalizedModel(model).to(DEVICE)


def run_attack(pil_image, model_choice, attack_choice, eps):
    if pil_image is None:
        return None, None, "No image provided", "No image provided", "—"

    norm_model = load_model(model_choice)
    norm_model.eval()

    img_t = TRANSFORM(pil_image.convert('RGB')).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = norm_model(img_t)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    clean_label = f"{CIFAR10_CLASSES[pred]}  ({probs[pred]*100:.1f}%)"
    target      = torch.tensor([pred]).to(DEVICE)

    if attack_choice == 'None':
        orig_pil   = tensor_to_pil(img_t)
        blank_pert = tensor_to_pil(torch.zeros_like(img_t))
        return orig_pil, blank_pert, clean_label, clean_label, "No attack applied"

    if attack_choice == 'FGSM':
        atk   = torchattacks.FGSM(norm_model, eps=eps)
        adv_t = atk(img_t, target)

    elif attack_choice == 'PGD':
        atk   = torchattacks.PGD(norm_model, eps=eps, alpha=eps / 4, steps=40, random_start=True)
        adv_t = atk(img_t, target)

    else:
        atk   = PatchAttack(norm_model, patch_size=56, max_iter=150, lr=0.05, device=DEVICE)
        adv_t = atk(img_t, target)

    adv_pil = tensor_to_pil(adv_t)

    with torch.no_grad():
        adv_logits = norm_model(adv_t)
        adv_probs  = torch.softmax(adv_logits, dim=1)[0]
        adv_pred   = adv_probs.argmax().item()

    adv_label = f"{CIFAR10_CLASSES[adv_pred]}  ({adv_probs[adv_pred]*100:.1f}%)"

    pert     = (adv_t - img_t).abs()
    pert_pil = tensor_to_pil((pert * 10).clamp(0, 1))

    status = "FOOLED" if pred != adv_pred else "Robust (prediction unchanged)"

    return adv_pil, pert_pil, clean_label, adv_label, status


with gr.Blocks(title="Adversarial Robustness Framework") as demo:
    gr.Markdown(
        "# Adversarial Robustness Framework\n"
        "Upload a CIFAR-10 style image and see how adversarial attacks change the model's prediction."
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp_image     = gr.Image(type='pil', label="Input Image")
            model_choice  = gr.Radio(['ViT (DeiT-Small)', 'ResNet-18'], value='ViT (DeiT-Small)', label="Model")
            attack_choice = gr.Radio(['None', 'FGSM', 'PGD', 'Patch'], value='None', label="Attack")
            eps_slider    = gr.Slider(0.01, 0.30, value=0.03, step=0.01, label="Epsilon (perturbation strength)")
            run_btn       = gr.Button("Run", variant="primary")

        with gr.Column(scale=1):
            out_adv    = gr.Image(label="Adversarial Image")
            out_pert   = gr.Image(label="Perturbation (amplified 10x)")
            clean_out  = gr.Textbox(label="Clean Prediction")
            adv_out    = gr.Textbox(label="Adversarial Prediction")
            status_out = gr.Textbox(label="Result")

    run_btn.click(
        run_attack,
        inputs=[inp_image, model_choice, attack_choice, eps_slider],
        outputs=[out_adv, out_pert, clean_out, adv_out, status_out],
    )

if __name__ == '__main__':
    demo.launch(share=False)
