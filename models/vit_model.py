import os
import timm
import torch


def load_vit(config, device):
    """Load DeiT-Small fine-tuned for CIFAR-10 (10 classes)."""
    arch       = config['models']['vit']['arch']
    pretrained = config['models']['vit']['pretrained']
    num_classes = config['models']['vit']['num_classes']
    checkpoint  = config['models']['vit'].get('checkpoint')

    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)

    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[ViT] Loaded checkpoint: {checkpoint}")
    elif checkpoint:
        print(f"[ViT] Checkpoint not found at {checkpoint}, using pretrained ImageNet weights")

    return model.to(device)
