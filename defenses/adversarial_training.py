import os
import torch
import torch.nn as nn
import torchattacks
from tqdm import tqdm

from data.loader import CIFAR10_MEAN, CIFAR10_STD


class _NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def _denorm(images, device):
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(device)
    return (images * std + mean).clamp(0, 1)


def _renorm(images, device):
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(device)
    return (images - mean) / std


def adversarial_train(model, train_loader, config, device, save_path=None):
    epochs  = config['training']['adv_training_epochs']
    lr      = config['training']['lr']
    adv_eps = config['training']['adv_eps']
    wd      = config['training']['weight_decay']

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = total_correct = total = 0

        norm_model = _NormalizedModel(model).to(device)
        atk = torchattacks.PGD(norm_model, eps=adv_eps, alpha=adv_eps / 4, steps=7)

        for images, labels in tqdm(train_loader, desc=f"Adv Train {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            images_01 = _denorm(images, device)

            norm_model.eval()
            model.eval()
            adv_01 = atk(images_01, labels)
            model.train()
            norm_model.train()

            adv_norm = _renorm(adv_01, device)

            optimizer.zero_grad()
            outputs = model(adv_norm)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * images.size(0)
            _, predicted   = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total         += images.size(0)

        scheduler.step()
        avg_loss = total_loss / total
        avg_acc  = total_correct / total
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}  Acc: {avg_acc:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved adversarially trained model -> {save_path}")

    return history
