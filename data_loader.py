import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)


class CIFAR100Custom(Dataset):
    def __init__(self, root, download=True, mode='train') -> None:
        super().__init__()
        torch.manual_seed(42)
        self.mode = mode
        
        if mode in ('train', 'val'):
            self.dataset = CIFAR100(
                root, train=True,
                transform=Compose([
                    ToTensor(),
                    Resize((224, 224)),
                ]),
                download=download
            )
            if mode == 'train':
                self.dataset, _ = random_split(self.dataset, [45000, 5000])
            
            else:
                _, self.dataset = random_split(self.dataset, [45000, 5000])

        elif mode == 'test':
            self.dataset = CIFAR100(
                root, train=False,
                transform=Compose([
                    ToTensor(),
                    Resize((224, 224)),
                ]),
                download=download
            )

        else:
            raise ValueError(f'check model: {mode}')
        
        self.length = len(self.dataset)
        self.transform = Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ])
        return
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.mode == 'train':
            image, label = self.dataset[index]
            image = self.transform(image)
        
        elif self.mode in ('val', 'test'):
            image, label = self.dataset[index]
        
        else:
            raise ValueError(f'check mode: {self.mode}')
        
        return dict(image=image.contiguous(), label=label)


def cifar100_loader(root, download):
    train_set = CIFAR100Custom(root, download, mode='train')
    val_set = CIFAR100Custom(root, download, mode='val')
    return train_set, val_set