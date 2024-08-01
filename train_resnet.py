from hydra import compose, initialize

from data_loader import cifar100_loader
from models.resnet import ResnetClassifier
from train import train


def main():
    with initialize(config_path='./configs', version_base='1.3.2'):
        config = compose(config_name='resnet50_ver1')
    
    train_loader, val_loader = cifar100_loader(**config.data, type='dataloader')
    train(ResnetClassifier, config, train_loader, val_loader)

    return


if __name__ == '__main__':
    main()