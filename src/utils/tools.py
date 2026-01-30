from statistics import mean

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import BaseModel


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


@torch.inference_mode()
def val_loop(model: BaseModel, val_loader: DataLoader, kind="val") -> dict[str, float]:
    device = model.device
    val_metrics = TensorDict(dict(), batch_size=(len(val_loader),), device=device)

    model.eval()
    for i, batch in enumerate(val_loader):
        batch: TensorDict
        batch = batch.to(device)

        batch_metrics = model.validate_batch(**batch)
        val_metrics[i] = batch_metrics

    val_metrics = {f"{kind}/{key}": value.item() for key, value in val_metrics.mean().to_dict().items()}

    return val_metrics


def pbar_finish(pbar: tqdm, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> dict[str, float]:
    metrics = train_metrics | val_metrics
    pbar.set_postfix(metrics)
    return metrics
