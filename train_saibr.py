import pandas as pd
import torch
from hydra import compose, initialize
from torch.utils.data import Dataset

from models import ImputationModel
from train import train


class CustomDataset(Dataset):
    def __init__(self, root, window_size, mask_prob, kind='train') -> None:
        super().__init__()
        assert kind in ('train', 'val', 'test')
        self.data = pd.read_csv(root)['OT'].values
        if kind == 'train':
            self.data = self.data[:int(0.7 * len(self.data))]
        
        elif kind == 'val':
            self.data = self.data[int(0.7 * len(self.data)):int(0.85 * len(self.data))]
        
        else:
            self.data = self.data[int(0.85 * len(self.data)):]
        self.window_size = window_size
        self.mask_prob = mask_prob
        self.length = self.data.shape[0] - window_size + 1
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        tensor = torch.FloatTensor(self.data[index:index+self.window_size])
        mask = (torch.rand(*tensor.size()) < self.mask_prob)

        tensor = tensor.view(-1, 1)
        mask = mask.view(-1, 1)
        
        return dict(x=tensor, mask=mask)
    

def load_dataset(config):
    train_set = CustomDataset(kind='train', **config.dataset)
    val_set = CustomDataset(kind='val', **config.dataset)

    return train_set, val_set


def main():
    with initialize(config_path='./configs', version_base='1.3.2'):
        config = compose(config_name='bid_rnn')

    train_set, val_set = load_dataset(config)
    train(ImputationModel, config, train_set, val_set)

    return


if __name__ == '__main__':
    main()