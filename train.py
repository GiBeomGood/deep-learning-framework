import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from models import BaseModel
from tools import EarlyStopping, mean, val_loop


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
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch:2d}", disable=disable_pbar)
    pbar_update = pbar.update
    train_metrics = {key: [] for key in model.train_keys}

    model.train()
    for batch in train_loader:
        batch: dict[str, Tensor]
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

        optimizer.zero_grad()
        batch_metrics: dict[str, Tensor] = model(**batch)
        loss = batch_metrics["loss"]
        loss.backward()
        optimizer.step()

        for key, value in batch_metrics.items():
            train_metrics[key].append(value.item())

        if pbar.n % check_unit == 0:
            pbar.set_postfix({key: mean(value_list) for key, value_list in train_metrics.items()})
        pbar_update()

    train_metrics = {f"train/{key}": mean(value_list) for key, value_list in train_metrics.items()}
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
    train_set: Dataset,
    val_set: Dataset,
):
    if config.wandb.do is True:
        run = wandb.init(
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            **config.wandb.kwargs,
        )

    train_loader = DataLoader(train_set, **config.dataloader.train)
    val_loader = DataLoader(val_set, **config.dataloader.val)
    optimizer = torch.optim.Adam(model.parameters(), **config.optimizer)

    early_stopping = EarlyStopping(**config.early_stopping)
    check_unit = int(len(train_loader) * config.check_prop)
    disable_pbar = config.disable_pbar

    for epoch in range(config.epochs):
        metrics = train_epoch(model, train_loader, val_loader, optimizer, epoch, check_unit, disable_pbar)

        if config.wandb.do is True:
            run.log(metrics, step=epoch)

        early_stop, improved = early_stopping.check(metrics)
        if early_stop is True:
            break

        if improved is True:
            torch.save(
                model.state_dict(),
                f"{config.model_save_path}/{config.model_name}-epoch{epoch:03d}.pt",
            )

    if config.wandb.do is True:
        run.finish()
    return
