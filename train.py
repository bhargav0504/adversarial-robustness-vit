import argparse
import os

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data.loader import get_loaders
from defenses.adversarial_training import adversarial_train
from models import load_resnet, load_vit


def fine_tune(model, train_loader, config, device, save_path):
    epochs = config['training']['epochs']
    lr     = config['training']['lr']
    wd     = config['training']['weight_decay']

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = total_correct = total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * images.size(0)
            _, predicted   = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total         += images.size(0)

        scheduler.step()
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {total_loss/total:.4f}  Acc: {total_correct/total:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved -> {save_path}\n")


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    train_loader, _ = get_loaders(config)

    if args.model in ('vit', 'both'):
        print("\n=== Fine-tuning ViT ===")
        vit = load_vit(config, device)
        fine_tune(vit, train_loader, config, device, './checkpoints/vit_cifar10.pth')

        if args.adv_train:
            print("\n=== Adversarially Training ViT ===")
            adversarial_train(vit, train_loader, config, device, './checkpoints/vit_cifar10_adv.pth')

    if args.model in ('resnet', 'both'):
        print("\n=== Fine-tuning ResNet-18 ===")
        resnet = load_resnet(config, device)
        fine_tune(resnet, train_loader, config, device, './checkpoints/resnet_cifar10.pth')

        if args.adv_train:
            print("\n=== Adversarially Training ResNet ===")
            adversarial_train(resnet, train_loader, config, device, './checkpoints/resnet_cifar10_adv.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--model', choices=['vit', 'resnet', 'both'], default='both')
    parser.add_argument('--adv-train', action='store_true')
    parser.add_argument('--device', default='auto')
    main(parser.parse_args())
