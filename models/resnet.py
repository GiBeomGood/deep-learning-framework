import torch
from torch import Tensor, nn

from .base_model import BaseModel


def conv3x3(inplanes, planes, stride):
    return nn.Conv2d(
        inplanes, planes,
        kernel_size=3, stride=stride,
        padding=1, bias=False
    )

def conv1x1(inplanes, planes, stride):
    return nn.Conv2d(
        inplanes, planes,
        kernel_size=1, stride=stride,
        padding=0, bias=False
    )


class SmallBlock(nn.Module):
    expansion: int = None
    def __init__(self):
        super().__init__()
        return


class BasicSmallBlock(SmallBlock):
    expansion = 1
    def __init__(self, inplanes, planes, stride) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            conv3x3(planes, planes, stride=1),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or inplanes != planes * self.expansion:
            self.identity = conv1x1(inplanes, planes*self.expansion, stride)
        else:
            self.identity = nn.Identity()
        return
    
    def forward(self, x) -> Tensor:
        output: Tensor = self.layer(x) + self.identity(x)
        return output.relu()

class BottleneckSmallBlock(SmallBlock):
    expansion = 4
    def __init__(self, inplanes, planes, stride) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            conv1x1(inplanes, planes, stride=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            conv3x3(planes, planes, stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            conv1x1(planes, planes*self.expansion, stride=1),
            nn.BatchNorm2d(planes*self.expansion),
        )

        if stride != 1 or inplanes != planes * self.expansion:
            self.identity = conv1x1(inplanes, planes*self.expansion, stride)
        else:
            self.identity = nn.Identity()
        return
    
    def forward(self, x) -> Tensor:
        output: Tensor = self.layer(x) + self.identity(x)
        return output.relu()


class BigBlock(nn.Module):
    def __init__(self, block: SmallBlock, inplanes, planes, stride, layers_count):
        super().__init__()
        expansion = block.expansion

        self.layers = [block(inplanes, planes, stride)]
        for _ in range(layers_count-1):
            self.layers.append(
                block(planes*expansion, planes, 1)
            )
        self.layers = nn.Sequential(*self.layers)
        return
    
    def forward(self, x) -> Tensor:
        return self.layers(x)


class Resnet(BaseModel):
    planes_list = [64, 128, 256, 512]
    layers_counts_dict = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }
    def __init__(self, model_num, output_dim=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        info_list = (
            self.planes_list,
            (1, 2, 2, 2),
            self.layers_counts_dict[model_num],
        )
        # planes, stride, layers_count

        if model_num in [18, 34]:
            block = BasicSmallBlock

        elif model_num in [50, 101, 152]:
            block = BottleneckSmallBlock

        else:
            raise ValueError(f'`model_num` should be one of [18, 34, 50, 101, 152]. got: {model_num}')

        self.layers = []
        inplanes = 64
        for planes, stride, layers_count in zip(*info_list):
            self.layers.append(
                BigBlock(block, inplanes, planes, stride, layers_count)
            )
            inplanes = planes * block.expansion
        self.layers = nn.Sequential(*self.layers)

        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(inplanes, output_dim),
        )

        return
    
    def get_output(self, x) -> Tensor:
        # initial layers
        output = self.conv1(x)
        output: Tensor = self.batchnorm1(output)
        output = self.maxpool(output.relu())

        # conv layers with residual connecttion
        output = self.layers(output)

        # final layers
        output = self.final_layer(output)
        
        return output
    
    @torch.no_grad()
    def predict(self, x) -> tuple[Tensor, Tensor]:
        output = self.get_output(x)

        return output, output.softmax(dim=1)
    
    def forward(self, image: Tensor, label: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError('define forward function')
    
    def validate_batch(self, image: Tensor, label: Tensor) -> dict[str, float]:
        raise NotImplementedError('define validate_batch function')


class ResnetClassifier(Resnet):
    train_keys = ('loss', )
    val_keys = ('loss', 'accuracy')

    def __init__(
            self, model_num=50, output_dim=100,
            loss_kwargs={}, val_loss_kwargs={},
    ):
        super().__init__(model_num, output_dim)
        self.criterion: nn.Module = nn.CrossEntropyLoss(**loss_kwargs)
        self.val_criterion: nn.Module = nn.CrossEntropyLoss(**val_loss_kwargs)
        self.accuracy = lambda label_pred, label: \
            (label_pred == label).sum()
        return
    
    def forward(self, image, label) -> dict[str, Tensor]:
        output = self.get_output(image)
        loss = self.criterion(output, label)
        return dict(loss=loss)
    
    @torch.no_grad()
    def validate_batch(self, image, label) -> dict[str, Tensor]:
        output, prob = self.predict(image)
        label_pred = prob.argmax(dim=1)

        loss = self.val_criterion(output, label).item()
        accuracy = self.accuracy(label_pred, label).item()

        return dict(zip(self.val_keys, (loss, accuracy)))