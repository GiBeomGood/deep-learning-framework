from math import sqrt

import torch
from torch import Tensor, nn

from .base_model import BaseModel


class EmbeddingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size):
        super().__init__()
        self.num_tokens = (img_size // patch_size) ** 2 + 1
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

        self.class_token = nn.Parameter(torch.randn(1, 1, out_channels))
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_tokens, out_channels))

        nn.init.normal_(self.class_token, mean=0, std=1e-6)
        nn.init.trunc_normal_(self.pos_encoding, mean=0, std=0.02)

        return

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = self.layer(x)  # (-1 x out_channels x H' x W')
        output = output.flatten(start_dim=2)  # (-1 x out_channels x num_tokens-1)
        output = output.permute(0, 2, 1)  # (-1 x num_tokens-1 x out_channels)

        class_token = self.class_token.repeat((x.size(0), 1, 1))  # (-1 x 1 x out_channels)
        # repeat makes incontiguous tensor contiguous
        output = torch.cat([output, class_token], dim=1)  # (-1 x num_tokens x out_channels)
        output += self.pos_encoding

        return output


class MultiheadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = input_dim // num_heads
        self.scaling_factor = sqrt(self.hidden_dim)
        self.layer_qkv = nn.Linear(input_dim, input_dim * 3, bias=False)
        self.linear_layer = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Dropout(dropout))
        return

    def forward(self, query: Tensor) -> Tensor:
        input_size = query.size()
        query = self.layer_qkv(query)  # (-1 x T x 3*input_dim)

        query = query.view(*input_size[:2], 3, self.num_heads, self.hidden_dim)  # (-1 x T x 3 x num_heads x hidden_dim)
        query = query.permute(2, 0, 3, 1, 4)  # (3 x -1 x num_heads x T x hidden_dim)
        key = query[1]
        value = query[2]
        query = query[0]

        attention: Tensor = query @ key.mT / self.scaling_factor  # (-1 x num_heads x T x T)
        attention = attention.softmax(dim=-1)  # (-1 x num_heads x T x T)

        output: Tensor = attention @ value  # (-1 x num_heads x T x hidden_dim)
        output = output.permute(0, 2, 1, 3).reshape(*input_size)  # (-1 x T x input_dim)
        output = self.linear_layer(output)

        return output


class Block(nn.Module):
    def __init__(self, input_dim, expansion, num_heads, dropout=0):
        super().__init__()
        self.attention_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            MultiheadSelfAttention(input_dim, num_heads, dropout),
        )
        self.layer_linear = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * expansion, input_dim),
            nn.Dropout(dropout),
        )

        return

    def forward(self, x: Tensor) -> Tensor:
        output = self.attention_layer(x) + x
        output = self.layer_linear(output) + output

        return output


class VisionTransformer(BaseModel):
    # in_channels -> hid_channels -> hid_channels x expansion -> num_classes
    def __init__(
        self,
        in_channels,
        hidden_channels,
        expansion,
        num_classes,
        image_size,
        patch_size,
        num_heads,
        num_blocks,
        dropout,
    ):
        assert hidden_channels % num_heads == 0, (
            "`hidden_dim` should be divided by `num_heads`. Check the parameter combinations."
        )
        super().__init__()

        self.layer_embedding = EmbeddingLayer(in_channels, hidden_channels, image_size, patch_size)
        self.blocks = nn.Sequential(*[Block(hidden_channels, expansion, num_heads, dropout) for _ in range(num_blocks)])
        self.layer_final = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, num_classes))

        return

    def get_output(self, x: Tensor) -> Tensor:
        output: Tensor = self.layer_embedding(x)  # (-1 x num_tokens x hidden_dim)
        output = self.blocks(output)  # (-1 x num_tokens x hidden_dim)
        output = output.mean(dim=1)  # (-1 x hidden_dim)
        output = self.layer_final(output)  # (-1 x output_dim)

        return output

    @torch.inference_mode()
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        output = self.get_output(x)

        return output, output.softmax(dim=1)

    def forward(self, image: Tensor, label: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError("define forward function")

    def validate_batch(self, image: Tensor, label: Tensor) -> dict[str, float]:
        raise NotImplementedError("define validate_batch function")


class ViTClassifier(VisionTransformer):
    train_keys = ("loss",)
    val_keys = ("loss", "accuracy")

    def __init__(
        self,
        vit_kwargs={},
        loss_kwargs={},
        val_loss_kwargs={},
    ):
        super().__init__()
        self.model = VisionTransformer(**vit_kwargs)

        self.criterion = nn.CrossEntropyLoss(**loss_kwargs)
        self.val_criterion = nn.CrossEntropyLoss(**val_loss_kwargs)
        self.accuracy = lambda label_pred, label: (label_pred == label).sum()
        return

    def forward(self, image: Tensor, label: Tensor) -> dict[str, Tensor]:
        output = self.model.get_output(image)
        loss = self.criterion(output, label)
        return dict(loss=loss)

    @torch.inference_mode()
    def validate_batch(self, image: Tensor, label: Tensor) -> dict[str, Tensor]:
        output, prob = self.model.predict(image)
        label_pred = prob.argmax(dim=1)

        loss = self.val_criterion(output, label).item()
        accuracy = self.accuracy(label_pred, label).item()

        return dict(zip(self.val_keys, (loss, accuracy)))
