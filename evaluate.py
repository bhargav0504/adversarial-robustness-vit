"""
evaluate.py — Benchmark clean and robust accuracy for all trained models.

Usage:
  python evaluate.py
  python evaluate.py --device cpu
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from data.loader import get_loaders
from evaluation.metrics import evaluate_model
from models import load_resnet, load_vit


def plot_results(all_results, save_dir):
    df = pd.DataFrame(all_results)
    attacks = ['clean_acc', 'fgsm_acc', 'pgd_acc', 'patch_acc']
    labels  = ['Clean', 'FGSM', 'PGD', 'Patch']

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.8 / len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        x = [j + i * width for j in range(len(attacks))]
        ax.bar(x, [row[a] * 100 for a in attacks], width, label=row['model'])

    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Adversarial Robustness: ViT vs ResNet on CIFAR-10')
    ax.set_xticks([j + (len(df) - 1) * width / 2 for j in range(len(attacks))])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'robustness_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {path}")


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    _, test_loader = get_loaders(
        config, n_eval_samples=config['evaluation']['n_samples']
    )

    os.makedirs(config['results_dir'], exist_ok=True)
    all_results = []

    # --- ViT (standard fine-tuned) ---
    config['models']['vit']['checkpoint'] = './checkpoints/vit_cifar10.pth'
    vit = load_vit(config, device)
    all_results.append(evaluate_model(vit, test_loader, config, device, "ViT (Standard)"))

    # --- ViT (adversarially trained) ---
    adv_ckpt = './checkpoints/vit_cifar10_adv.pth'
    if os.path.exists(adv_ckpt):
        config['models']['vit']['checkpoint'] = adv_ckpt
        vit_adv = load_vit(config, device)
        all_results.append(
            evaluate_model(vit_adv, test_loader, config, device, "ViT (Adv. Trained)")
        )

    # --- ResNet-18 (standard fine-tuned) ---
    config['models']['resnet']['checkpoint'] = './checkpoints/resnet_cifar10.pth'
    resnet = load_resnet(config, device)
    all_results.append(
        evaluate_model(resnet, test_loader, config, device, "ResNet18 (Standard)")
    )

    # --- ResNet-18 (adversarially trained) ---
    adv_ckpt_r = './checkpoints/resnet_cifar10_adv.pth'
    if os.path.exists(adv_ckpt_r):
        config['models']['resnet']['checkpoint'] = adv_ckpt_r
        resnet_adv = load_resnet(config, device)
        all_results.append(
            evaluate_model(resnet_adv, test_loader, config, device, "ResNet18 (Adv. Trained)")
        )

    # Save JSON
    json_path = os.path.join(config['results_dir'], 'results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {json_path}")

    # Plot
    plot_results(all_results, config['results_dir'])

    # Summary table
    df = pd.DataFrame(all_results)
    for col in ['clean_acc', 'fgsm_acc', 'pgd_acc', 'patch_acc']:
        df[col] = (df[col] * 100).round(2)
    df.columns = ['Model', 'Clean %', 'FGSM %', 'PGD %', 'Patch %']
    print(f"\n{df.to_string(index=False)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--device', default='auto')
    main(parser.parse_args())
