"""
realworld/demo.py — Real-world adversarial robustness demo.

Works on ANY real photo (humans, animals, objects, vehicles, etc.)
using ImageNet-pretrained models — no training required.

Run from the project root:
    python realworld/demo.py

Then open http://localhost:7860
"""
import os
import sys
import torch
import torchattacks

# Fix Gradio permission error on Windows — redirect temp uploads to a local folder
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GRADIO_TMP   = os.path.join(_PROJECT_ROOT, '.gradio_tmp')
os.makedirs(_GRADIO_TMP, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = _GRADIO_TMP

sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
from PIL import Image

from realworld.loader import pil_to_tensor, tensor_to_pil
from realworld.models import load_vit_imagenet, load_resnet_imagenet
from realworld.classify import predict_topk, format_predictions

# ── Setup ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-load both models once at startup
print(f"Loading models on {DEVICE}...")
_MODELS = {
    'ViT (DeiT-Small)': load_vit_imagenet(DEVICE),
    'ResNet-18':         load_resnet_imagenet(DEVICE),
}
print("Models ready.")


# ── Core inference ─────────────────────────────────────────────────────────────
def run(pil_image, model_name, attack_name, eps, top_k):
    if pil_image is None:
        return None, None, "No image uploaded.", "No image uploaded.", "—", "—"

    model = _MODELS[model_name]
    img_t = pil_to_tensor(pil_image).to(DEVICE)

    # ── Clean prediction ──────────────────────────────────────────────────────
    clean_preds = predict_topk(model, img_t, k=int(top_k))
    clean_top1  = clean_preds[0]
    clean_summary = (
        f"{clean_top1['class']}\n"
        f"Confidence: {clean_top1['confidence']}%\n"
        f"Category:   {clean_top1['category']}"
    )
    clean_topk_text = format_predictions(clean_preds)

    if attack_name == 'None':
        return (
            pil_image,
            None,
            clean_summary,
            "No attack selected.",
            clean_topk_text,
            "—",
        )

    # ── Generate adversarial example ──────────────────────────────────────────
    label = torch.tensor([clean_top1['rank'] - 1 +
                          list(range(1000))[clean_preds[0]['rank'] - 1]]).to(DEVICE)

    # Use the actual ImageNet index of the top-1 prediction
    import torch.nn.functional as F
    with torch.no_grad():
        logits = model(img_t)
    top1_idx = logits.argmax(dim=1)

    if attack_name == 'FGSM':
        atk = torchattacks.FGSM(model, eps=eps)
    elif attack_name == 'PGD':
        atk = torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=40)
    else:  # Patch
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from attacks.patch_attack import PatchAttack
        atk = PatchAttack(model, patch_size=56, max_iter=50, device=DEVICE)
        adv_t = atk(img_t, top1_idx)
        adv_t = adv_t.to(DEVICE)

    if attack_name in ('FGSM', 'PGD'):
        adv_t = atk(img_t, top1_idx)

    adv_pil  = tensor_to_pil(adv_t)

    # ── Adversarial prediction ────────────────────────────────────────────────
    adv_preds = predict_topk(model, adv_t, k=int(top_k))
    adv_top1  = adv_preds[0]
    adv_summary = (
        f"{adv_top1['class']}\n"
        f"Confidence: {adv_top1['confidence']}%\n"
        f"Category:   {adv_top1['category']}"
    )
    adv_topk_text = format_predictions(adv_preds)

    status = (
        "FOOLED ✗  — prediction changed!"
        if clean_top1['class'] != adv_top1['class']
        else "Robust ✓  — prediction unchanged"
    )

    return pil_image, adv_pil, clean_summary, adv_summary, clean_topk_text, adv_topk_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Real-World Adversarial Robustness") as demo:
    gr.Markdown(
        "# Real-World Adversarial Robustness Demo\n"
        "Upload **any real photo** — people, animals, objects, vehicles, anything.\n"
        "The model uses **ImageNet-1k** (1000 classes) so no training is needed.\n"
        "Apply an adversarial attack and see how the prediction changes."
    )

    with gr.Row():
        # ── Left: controls ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            inp_image    = gr.Image(type='pil', label="Upload Any Real Image")
            model_choice = gr.Radio(
                ['ViT (DeiT-Small)', 'ResNet-18'],
                value='ViT (DeiT-Small)', label="Model"
            )
            attack_choice = gr.Radio(
                ['None', 'FGSM', 'PGD', 'Patch'],
                value='None', label="Attack"
            )
            eps_slider = gr.Slider(
                0.001, 0.1, value=0.02, step=0.001,
                label="Epsilon  (perturbation strength)"
            )
            topk_slider = gr.Slider(
                1, 10, value=5, step=1, label="Show Top-K predictions"
            )
            run_btn = gr.Button("Analyse", variant="primary")

        # ── Right: outputs ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Row():
                out_original = gr.Image(label="Original Image")
                out_adv      = gr.Image(label="Adversarial Image")
            with gr.Row():
                out_clean_summary = gr.Textbox(label="Clean Prediction", lines=3)
                out_adv_summary   = gr.Textbox(label="Adversarial Prediction", lines=3)
            out_clean_topk = gr.Textbox(label="Clean  — Top-K", lines=6, max_lines=10)
            out_adv_topk   = gr.Textbox(label="Adversarial  — Top-K", lines=6, max_lines=10)

    run_btn.click(
        run,
        inputs=[inp_image, model_choice, attack_choice, eps_slider, topk_slider],
        outputs=[out_original, out_adv,
                 out_clean_summary, out_adv_summary,
                 out_clean_topk, out_adv_topk],
    )

    gr.Markdown(
        "---\n"
        "**Categories:** Human · Animal · Plant · Non-Living Object  \n"
        "**Models:** ImageNet-1k pretrained — no fine-tuning needed  \n"
        "**Note:** This is a separate module from the CIFAR-10 pipeline. "
        "Delete the `realworld/` folder to remove it entirely."
    )

if __name__ == '__main__':
    print(f"\nRunning on: {DEVICE}")
    demo.launch(share=False)
