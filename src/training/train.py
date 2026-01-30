from pathlib import Path

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import BaseModel
from src.utils.tools import EarlyStopping, val_loop


def train_epoch(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    check_unit: int,
    disable_pbar: bool,
) -> dict[str, float]:
    device = model.device
    train_metrics = TensorDict(dict(), batch_size=(len(train_loader),), device=device)
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch:2d}", disable=disable_pbar)

    model.train()
    for i, batch in enumerate(train_loader):
        batch: TensorDict
        # batch = batch.to(device, non_blocking=True)
        batch = batch.to(device)

        optimizer.zero_grad()
        output: TensorDict = model(**batch)
        output["loss"].backward()
        optimizer.step()

        train_metrics[i] = output["metrics"]

        if pbar.n % check_unit == 0:
            pbar.set_postfix(train_metrics[: i + 1].mean().to_dict())
        pbar.update()

    train_metrics = {f"train/{key}": value.item() for key, value in train_metrics.mean().to_dict().items()}
    val_metrics = val_loop(model, val_loader, kind="val")
    metrics = train_metrics | val_metrics

    metrics_str = ", ".join(f"{key}={metrics:5.3f}" for key, metrics in metrics.items())
    pbar.set_postfix_str(metrics_str)

    if disable_pbar is True:
        print(f"epoch {epoch:3d}: {metrics_str}")

    return metrics


def train(
    model: BaseModel,
    config: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model_compiled = torch.compile(model, mode="reduce-overhead", disable=(not config.compile))
    Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model_compiled.parameters(), **config.optimizer)

    early_stopping = EarlyStopping(**config.early_stopping)
    check_unit = int(len(train_loader) * config.check_prop)

    for epoch in range(config.epochs):
        metrics = train_epoch(
            model_compiled,
            train_loader,
            val_loader,
            optimizer,
            epoch,
            check_unit,
            config.disable_pbar,
        )

        early_stop, improved = early_stopping.check(metrics)
        if early_stop is True:
            break

        if improved is True:
            torch.save(model.state_dict(), config.model_save_path)

    return
