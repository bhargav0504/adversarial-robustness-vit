import torch
from tqdm import tqdm

from attacks.fgsm import get_fgsm_attack
from attacks.pgd import get_pgd_attack
from attacks.patch_attack import PatchAttack


def compute_accuracy(model, loader, device, desc="Clean accuracy"):
    """Evaluate clean (unattacked) accuracy."""
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            _, predicted = model(images).max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

    return correct / total


def compute_robust_accuracy(model, loader, attack, device, desc="Robust accuracy"):
    """Evaluate accuracy under a given attack."""
    model.eval()
    correct = total = 0

    for images, labels in tqdm(loader, desc=desc):
        images, labels = images.to(device), labels.to(device)
        adv_images = attack(images, labels)

        with torch.no_grad():
            _, predicted = model(adv_images).max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

    return correct / total


def evaluate_model(model, loader, config, device, model_name="Model"):
    """
    Run full evaluation: clean accuracy + FGSM + PGD + Patch robustness.

    Returns a dict with keys:
        model, clean_acc, fgsm_acc, pgd_acc, patch_acc
    """
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    results = {'model': model_name}

    results['clean_acc'] = compute_accuracy(model, loader, device, "  Clean")

    fgsm = get_fgsm_attack(model, config)
    results['fgsm_acc'] = compute_robust_accuracy(
        model, loader, fgsm, device, "  FGSM"
    )

    pgd = get_pgd_attack(model, config)
    results['pgd_acc'] = compute_robust_accuracy(
        model, loader, pgd, device, "  PGD"
    )

    patch_atk = PatchAttack(
        model,
        patch_size=config['attacks']['patch']['patch_size'],
        max_iter=config['attacks']['patch']['max_iter'],
        device=device,
    )
    results['patch_acc'] = compute_robust_accuracy(
        model, loader, patch_atk, device, "  Patch"
    )

    print(f"\n  Clean: {results['clean_acc']*100:.2f}%  "
          f"FGSM: {results['fgsm_acc']*100:.2f}%  "
          f"PGD: {results['pgd_acc']*100:.2f}%  "
          f"Patch: {results['patch_acc']*100:.2f}%")
    return results
