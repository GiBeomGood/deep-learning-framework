from pathlib import Path

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.utils.data import Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CustomDataset(CIFAR10):
    def __init__(self, root: str, train: bool, transform=None, target_transform=None, download: bool = True, **kwargs):
        super().__init__(root, train, transform, target_transform, download)
        return

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        batch = TensorDict(
            {
                "image": image.float(),
                "target": torch.as_tensor(target, dtype=torch.long),
            },
            batch_size=(),
        )
        return batch


def load_cifar10(config: DictConfig):
    path = Path(config.root)
    path.mkdir(parents=True, exist_ok=True)

    transform_train = transforms.Compose(
        [
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    generator = torch.Generator().manual_seed(config.seed)

    train_set = CustomDataset(train=True, transform=transform_train, **config)
    train_indices, val_indices = random_split(range(len(train_set)), lengths=(0.8, 0.2), generator=generator)
    train_set = Subset(train_set, train_indices)

    val_set = CustomDataset(train=True, transform=transform_test, **config)
    val_set = Subset(val_set, val_indices)

    test_set = CustomDataset(train=False, transform=transform_test, **config)

    return train_set, val_set, test_set
