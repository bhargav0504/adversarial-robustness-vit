import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_transforms(image_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_loaders(config, n_eval_samples=None):
    image_size = config['data']['image_size']
    data_dir   = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']

    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_transforms(image_size, train=True)
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=get_transforms(image_size, train=False)
    )

    if n_eval_samples is not None:
        indices = np.random.choice(len(test_dataset), n_eval_samples, replace=False)
        test_dataset = Subset(test_dataset, indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def denormalize(tensor):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1).to(tensor.device)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)
