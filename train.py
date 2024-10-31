from statistics import mean

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from models import BaseModel
from tools import EarlyStopping, pbar_finish, val_loop


def train_loop(
    model: BaseModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    check_unit: int,
) -> tuple[tqdm, dict[str, float]]:
    device = next(model.parameters()).device
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch:2d}")
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
            pbar.set_postfix(
                {key: mean(value_list) for key, value_list in train_metrics.items()}
            )
        pbar_update()

    train_metrics = {
        f"train {key}": mean(value_list) for key, value_list in train_metrics.items()
    }

    return pbar, train_metrics


def train(
    model: type[BaseModel],
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
    early_stopping = EarlyStopping(**config.early_stopping)
    check_unit = int(len(train_loader) * config.check_prop)

    model = model(**config.model).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), **config.optimizer)

    for epoch in range(config.epochs):
        pbar, train_metrics = train_loop(
            model, train_loader, optimizer, epoch, check_unit
        )
        val_metrics = val_loop(model, val_loader)
        postfix = pbar_finish(pbar, train_metrics, val_metrics)

        if config.wandb.do is True:
            run.log(postfix, step=epoch)

        early_stop, improved = early_stopping.check(postfix)
        if early_stop is True:
            break

        if improved is True:
            torch.save(
                model.state_dict(),
                f"{config.model_save_path}/{config.model_name}_epoch{epoch:2d}.pth",
            )

    if config.wandb.do is True:
        run.finish()
    return
