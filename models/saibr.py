from math import sqrt

import torch
from torch import Tensor, nn

from . import BaseModel


class BidRnn(nn.Module):
    def __init__(self, hid_dim, n_layers=1):
        super().__init__()
        self.hid_dim = hid_dim
        self.gru_forward = nn.GRU(1, hid_dim, n_layers)
        self.gru_backward = nn.GRU(1, hid_dim, n_layers)
        self.predictor = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1),
            nn.Tanh(),
        )
        return

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # (-1 x T x 1), (-1 x T x 1) -> (-1 x T x m)
        h_stack_forward = []
        h_stack_backward = []
        x = x.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        h = self._init_gru(x)  # (2 x -1 x m)

        for time_step in range(x.size(0)):
            x_temp = torch.stack([x[time_step], x[-1-time_step]], dim=0)  # (2 x -1 x 1)
            mask_temp = torch.stack([mask[time_step], mask[-1-time_step]], dim=0)  # (2 x -1 x 1)

            x_temp = x_temp * ~mask_temp + self.predictor(h) * mask_temp  # (2 x -1 x 1)
            h_1, _ = self.gru_forward(x_temp[[0]], h[[0]])  # (1 x -1 x m)
            h_2, _ = self.gru_backward(x_temp[[1]], h[[1]])  # (1 x -1 x m)
            
            h = torch.cat([h_1, h_2], dim=0)  # (2 x -1 x m)
            h_stack_forward.append(h_1)  # (1 x -1 x m)
            h_stack_backward.append(h_2)  # (1 x -1 x m)
        
        h_stack_forward = torch.cat(h_stack_forward, dim=0)  # (T x -1 x m)
        h_stack_backward = torch.cat(h_stack_backward[::-1], dim=0)  # (T x -1 x m)
        h_stack = torch.cat([h_stack_forward, h_stack_backward], dim=2)  # (T x -1 x 2m)
        h_stack = h_stack.transpose(0, 1).contiguous()  # (-1 x T x 2m)
        return h_stack
    
    def _init_gru(self, x: Tensor):
        return torch.zeros(2, x.size(1), self.hid_dim).to(x.device)


class SeqAtt(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.linear_q = nn.Linear(input_dim, input_dim, bias=False)
        self.linear_k = nn.Linear(input_dim, input_dim, bias=False)
        self.linear_v = nn.Linear(input_dim, input_dim, bias=False)
        self.num_heads = num_heads
        self.hidden_dim = input_dim // num_heads
        self.scaling_factor = sqrt(self.hidden_dim)
        return
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # (-1 x T x 2m) -> (-1 x T x 2m)
        input_size = query.size()

        query = self.linear_q(query)  # (-1 x T x 2m)
        key = self.linear_k(key)  # (-1 x T x 2m)
        value = self.linear_v(value)  # (-1 x T x 2m)

        query = query.view(*input_size[:2], self.num_heads, self.hidden_dim)
        key = key.view(*input_size[:2], self.num_heads, self.hidden_dim)
        value = value.view(*input_size[:2], self.num_heads, self.hidden_dim)

        attention: Tensor = query @ key.mT / self.scaling_factor  # (-1 x num_heads x T x T)
        attention = attention.softmax(dim=-1)  # (-1 x num_heads x T x T)

        output: Tensor = attention @ value  # (-1 x num_heads x T x hidden_dim)
        output = output.permute(0, 2, 1, 3).contiguous().view(*input_size)  # (-1 x T x input_dim)
        return output


class ImputationModel(BaseModel):
    train_keys = ('loss', )
    val_keys = ('loss', 'mae')

    def __init__(
            self, hid_dim, n_layers=1, num_heads=1,
            loss_kwargs={}, val_loss_kwargs={},
        ):
        super().__init__()
        self.bid_rnn = BidRnn(hid_dim, n_layers=n_layers)
        self.seq_att = SeqAtt(2*hid_dim, num_heads=num_heads)
        self.final_layer = nn.Sequential(
            nn.Linear(2*hid_dim, 2*hid_dim),
            nn.ReLU(),
            nn.Linear(2*hid_dim, 1),
        )

        self.criterion = nn.MSELoss(**loss_kwargs)
        self.val_criterion = nn.MSELoss(**val_loss_kwargs)
        return
    
    def get_output(self, x, mask) -> Tensor:
        # (-1 x T x 1), (-1 x T x 1) -> (-1 x T x 2m)
        output = self.bid_rnn(x, mask)  # (-1 x T x 2m)
        output = self.seq_att(output, output, output)  # (-1 x T x 2m)
        output = self.final_layer(output)  # (-1 x T x 1)
        output[~mask] = x[~mask]

        return output
    
    def forward(self, x, mask) -> dict[str, Tensor]:
        output = self.get_output(x, mask)
        loss: Tensor = self.criterion(output[mask], x[mask])
        return dict(loss=loss)
    
    @torch.no_grad()
    def validate_batch(self, x: Tensor, mask: Tensor) -> dict[str, float]:
        output: Tensor = self.get_output(x, mask)
        x = x[mask]
        output = output[mask]
        
        loss: Tensor = self.val_criterion(output, x) / mask.sum() * mask.size(0)
        mae = (output-x).abs().sum() / mask.sum() * mask.size(0)

        loss = loss.item()
        mae = mae.item()
        
        results = dict(zip(self.val_keys, (loss, mae)))

        return results