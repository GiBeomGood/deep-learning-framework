import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CustomCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index) -> dict[str, torch.Tensor | int]:
        image, target = super().__getitem__(index)
        return dict(x=image, target=target)


def load_cifar10(root, kind: str, download=False, seed=42):
    if kind == "train":
        transform = transforms.Compose(
            [
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = CustomCIFAR10(root, train=True, transform=transform, download=download)
        dataset = random_split(dataset, lengths=[40000, 10000], generator=torch.Generator().manual_seed(seed))[0]

    elif kind == "val":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = CustomCIFAR10(root, train=True, transform=transform, download=download)
        dataset = random_split(dataset, lengths=[40000, 10000], generator=torch.Generator().manual_seed(seed))[1]

    elif kind == "test":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = CustomCIFAR10(root, train=False, transform=transform, download=download)

    return dataset


if __name__ == "__main__":
    for kind in ("train", "val", "test"):
        dataset = load_cifar10(root="./data", kind=kind, download=False)
        print(dataset[0]["x"].size())
        print(dataset[0]["target"])
        print(len(dataset))
