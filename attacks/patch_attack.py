import torch
import torch.nn as nn
import numpy as np


class PatchAttack:
    """
    Adversarial Patch Attack.

    Optimizes a rectangular patch placed at a fixed location so the model
    misclassifies the image regardless of background content.
    Expects images in [0, 1].
    """

    def __init__(self, model, patch_size=32, max_iter=100, lr=0.05, device='cpu'):
        self.model      = model
        self.patch_size = patch_size
        self.max_iter   = max_iter
        self.lr         = lr
        self.device     = device
        self.criterion  = nn.CrossEntropyLoss()

    def _apply(self, images, patch, x, y, ps):
        adv = images.clone()
        adv[:, :, y:y + ps, x:x + ps] = patch
        return adv

    def __call__(self, images, labels):
        images = images.detach().to(self.device)
        labels = labels.to(self.device)
        B, C, H, W = images.shape

        ps = min(self.patch_size, H, W)
        x, y = 0, 0

        patch = torch.rand(B, C, ps, ps, device=self.device)

        self.model.eval()

        for _ in range(self.max_iter):
            patch_var  = patch.clone().detach().requires_grad_(True)
            adv_images = self._apply(images, patch_var, x, y, ps)

            outputs = self.model(adv_images)
            loss    = self.criterion(outputs, labels)
            loss.backward()

            with torch.no_grad():
                patch = patch_var + self.lr * patch_var.grad.sign()
                patch = patch.clamp(0, 1)

        return self._apply(images, patch.detach(), x, y, ps).detach()
