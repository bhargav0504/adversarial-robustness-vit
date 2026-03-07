import os
import torch
import torch.nn as nn
from torchvision import models


def load_resnet(config, device):
    """Load ResNet-18 fine-tuned for CIFAR-10 (10 classes)."""
    arch        = config['models']['resnet']['arch']
    pretrained  = config['models']['resnet']['pretrained']
    num_classes = config['models']['resnet']['num_classes']
    checkpoint  = config['models']['resnet'].get('checkpoint')

    weights = 'IMAGENET1K_V1' if pretrained else None
    model = getattr(models, arch)(weights=weights)
    # Replace final FC layer for 10 CIFAR classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[ResNet] Loaded checkpoint: {checkpoint}")
    elif checkpoint:
        print(f"[ResNet] Checkpoint not found at {checkpoint}, using pretrained ImageNet weights")

    return model.to(device)
