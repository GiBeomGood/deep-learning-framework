import torch
import torch.nn.functional as F  # noqa: N812
from tensordict import TensorDict
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from .base_model import BaseModel


class ResNetClassifier(BaseModel):
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # (2048, 10)
        return

    def get_output(self, image: Tensor) -> Tensor:
        return self.resnet(image)

    def forward(self, image: Tensor, target: Tensor) -> TensorDict:
        output = self.get_output(image)  # (-1, C)
        loss = F.cross_entropy(output, target)
        output_dict = TensorDict(
            {
                "loss": loss,
                "metrics": {
                    "loss": loss.detach(),
                },
            },
            batch_size=(),
        )
        return output_dict

    @torch.inference_mode()
    def validate_batch(self, image: Tensor, target: Tensor) -> TensorDict:
        output = self.get_output(image)  # (-1, C)
        label_pred = output.argmax(1)  # (-1)

        loss = F.cross_entropy(output, target)
        acc = target.eq(label_pred).float().mean()
        output_dict = TensorDict(
            {
                "loss": loss,
                "acc": acc,
            },
            batch_size=(),
        )
        return output_dict
