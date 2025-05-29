from statistics import mean

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BaseModel


class EarlyStopping:
    def __init__(self, val_key: str, tolerance=5, higher_better=True):
        self.deviation = 0
        self.val_key = val_key
        self.tolerance = tolerance
        self.higher_better = higher_better

        if higher_better is True:
            self.best_standard = float("-inf")

        else:
            self.best_standard = float("inf")

        return

    def check(self, postfix: dict[str, float]) -> tuple[bool, bool]:
        early_stop = False
        standard = postfix[self.val_key]
        improved = self.improvement_check(standard)

        if improved is True:
            self.deviation = 0
            self.best_standard = standard

        else:
            self.deviation += 1
            if self.deviation > self.tolerance:
                early_stop = True

        return early_stop, improved

    def improvement_check(self, standard: float) -> bool:
        if self.higher_better is True:
            return standard > self.best_standard

        else:
            return standard < self.best_standard


@torch.no_grad()
def val_loop(model: BaseModel, val_loader: DataLoader, kind="val") -> dict[str, float]:
    device = model.device
    val_metrics = {key: [] for key in model.val_keys}

    model.eval()
    for batch in val_loader:
        batch: dict[str, Tensor]
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

        batch_metrics = model.validate_batch(**batch)
        for key, value in batch_metrics.items():
            val_metrics[key].append(value)

    val_metrics = {f"{kind}/{key}": mean(results) for key, results in val_metrics.items()}

    return val_metrics
