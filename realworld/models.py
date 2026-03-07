"""
realworld/models.py
Load ImageNet-pretrained ViT and ResNet — no fine-tuning needed.
These already know 1000 real-world categories.
"""
import torch
import timm
from torchvision import models


def load_vit_imagenet(device: torch.device):
    """DeiT-Small pretrained on ImageNet-1k (1000 classes)."""
    model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=1000)
    model.eval()
    return model.to(device)


def load_resnet_imagenet(device: torch.device):
    """ResNet-18 pretrained on ImageNet-1k (1000 classes)."""
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval()
    return model.to(device)
