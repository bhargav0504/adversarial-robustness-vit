import torch
import torch.nn as nn
import numpy as np


class PatchAttack:
    """
    Adversarial Patch Attack.

    Optimizes a small rectangular patch that, when placed at a random
    location in the image, causes the model to misclassify.
    Uses iterative sign-gradient updates (similar to FGSM per-step).
    """

    def __init__(self, model, patch_size=32, max_iter=100, device='cpu'):
        self.model     = model
        self.patch_size = patch_size
        self.max_iter  = max_iter
        self.device    = device
        self.criterion = nn.CrossEntropyLoss()

    def _apply(self, images, patch, x, y):
        """Apply patch tensor to images at position (x, y)."""
        adv = images.clone()
        adv[:, :, y:y + self.patch_size, x:x + self.patch_size] = patch
        return adv

    def __call__(self, images, labels):
        B, C, H, W = images.shape
        images = images.detach().to(self.device)
        labels = labels.to(self.device)

        ps = self.patch_size
        # Clamp patch size to image dimensions
        ps = min(ps, H, W)

        # Random patch location
        x = np.random.randint(0, max(1, W - ps))
        y = np.random.randint(0, max(1, H - ps))

        # Initialize patch with values from original image region
        patch = images[:, :, y:y + ps, x:x + ps].clone()

        self.model.eval()

        for _ in range(self.max_iter):
            patch_var = patch.clone().detach().requires_grad_(True)
            adv_images = self._apply(images, patch_var, x, y)

            outputs = self.model(adv_images)
            # Maximise loss (untargeted attack)
            loss = -self.criterion(outputs, labels)
            loss.backward()

            with torch.no_grad():
                # FGSM-style step
                patch = patch - 0.05 * patch_var.grad.sign()
                patch = patch.clamp(0, 1)

        return self._apply(images, patch.detach(), x, y).detach()
