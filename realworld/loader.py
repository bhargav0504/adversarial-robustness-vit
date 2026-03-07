"""
realworld/loader.py
Preprocessing for arbitrary real-world images (any resolution, any content).
Uses ImageNet normalization (different from CIFAR-10).
"""
import torch
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalised (1, 3, 224, 224) tensor."""
    return TRANSFORM(pil_image.convert('RGB')).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalised (1, C, H, W) or (C, H, W) tensor back to PIL."""
    import numpy as np
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.squeeze(0).cpu() * std + mean).clamp(0, 1)
    return Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
