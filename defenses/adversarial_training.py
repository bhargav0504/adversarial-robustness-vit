import os
import torch
import torch.nn as nn
import torchattacks
from tqdm import tqdm


def adversarial_train(model, train_loader, config, device, save_path=None):
    """
    PGD-based adversarial training.

    For each mini-batch the model sees adversarial examples (generated with
    a fast 7-step PGD) instead of clean images. This is the AT-PGD method
    from Madry et al. (2018), the standard baseline for certified robustness.
    """
    epochs     = config['training']['adv_training_epochs']
    lr         = config['training']['lr']
    adv_eps    = config['training']['adv_eps']
    wd         = config['training']['weight_decay']

    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion  = nn.CrossEntropyLoss()
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = total_correct = total = 0

        # Build attack fresh each epoch so it picks up any model changes
        atk = torchattacks.PGD(
            model, eps=adv_eps, alpha=adv_eps / 4, steps=7
        )

        for images, labels in tqdm(train_loader, desc=f"Adv Train {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial images (model set to eval inside torchattacks)
            model.eval()
            adv_images = atk(images, labels)
            model.train()

            optimizer.zero_grad()
            outputs = model(adv_images)
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
        print(f"Saved adversarially trained model → {save_path}")

    return history
