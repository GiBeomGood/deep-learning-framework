import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.cifar10 import load_cifar10
from src.models import ResNetClassifier
from src.training.train import train


def main():
    torch.set_float32_matmul_precision("high")

    config = OmegaConf.load("configs/resnet50.yaml")
    train_set, val_set, test_set = load_cifar10(config.dataset)

    train_loader = DataLoader(train_set, **config.dataloader.train, collate_fn=torch.stack)
    val_loader = DataLoader(val_set, **config.dataloader.val, collate_fn=torch.stack)
    del test_set
    # test_loader = DataLoader(test_set, **config.dataloader.test)

    model = ResNetClassifier(num_classes=10).to(0)
    train(model, config, train_loader, val_loader)

    return


if __name__ == "__main__":
    main()
