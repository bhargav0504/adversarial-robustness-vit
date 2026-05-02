import torch
import torch.nn as nn
from tqdm import tqdm

from attacks.fgsm import get_fgsm_attack
from attacks.pgd import get_pgd_attack
from attacks.patch_attack import PatchAttack
from data.loader import CIFAR10_MEAN, CIFAR10_STD


class NormalizedModel(nn.Module):
    """Wraps model with CIFAR-10 normalization so attacks work in [0, 1]."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std',  std)

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def _denorm(images, device):
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(device)
    return (images * std + mean).clamp(0, 1)


def compute_accuracy(model, loader, device, desc="Clean accuracy"):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            _, predicted = model(images).max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

    return correct / total


def compute_robust_accuracy(norm_model, raw_model, loader, attack_fn, device, desc="Robust accuracy"):
    norm_model.eval()
    raw_model.eval()
    correct = total = 0

    atk = attack_fn(norm_model)

    for images, labels in tqdm(loader, desc=desc):
        images, labels = images.to(device), labels.to(device)
        images_01 = _denorm(images, device)
        adv_01    = atk(images_01, labels)
        with torch.no_grad():
            _, predicted = norm_model(adv_01).max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

    return correct / total


def evaluate_model(model, loader, config, device, model_name="Model"):
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    norm_model = NormalizedModel(model).to(device)
    results = {'model': model_name}

    results['clean_acc'] = compute_accuracy(model, loader, device, "  Clean")

    eps   = config['attacks']['fgsm']['eps']
    pgd_a = config['attacks']['pgd']['alpha']
    pgd_s = config['attacks']['pgd']['steps']
    ps    = config['attacks']['patch']['patch_size']
    mi    = config['attacks']['patch']['max_iter']

    results['fgsm_acc'] = compute_robust_accuracy(
        norm_model, model, loader,
        lambda m: get_fgsm_attack(m, config),
        device, "  FGSM"
    )

    results['pgd_acc'] = compute_robust_accuracy(
        norm_model, model, loader,
        lambda m: get_pgd_attack(m, config),
        device, "  PGD"
    )

    results['patch_acc'] = compute_robust_accuracy(
        norm_model, model, loader,
        lambda m: PatchAttack(m, patch_size=ps, max_iter=mi, device=device),
        device, "  Patch"
    )

    print(f"\n  Clean: {results['clean_acc']*100:.2f}%  "
          f"FGSM: {results['fgsm_acc']*100:.2f}%  "
          f"PGD: {results['pgd_acc']*100:.2f}%  "
          f"Patch: {results['patch_acc']*100:.2f}%")
    return results
