---
name: deep-learning
description: >
  PyTorch deep learning project structure and conventions. Use this skill whenever
  the user is writing or organizing deep learning code in Python — model definitions,
  training loops, validation, early stopping, datasets, configs, or any task that
  involves structuring a DL project. Trigger even if the user only mentions one
  component (e.g., "write a training loop", "add early stopping", "define a model").
  This skill does NOT duplicate python-core; it builds on top of it for DL-specific
  structure.
---

# Deep Learning Project Structure

This skill defines how to organize and write PyTorch deep learning code in this project.
It covers structure and conventions only — not implementation details inside functions.
Follow `python-core` for naming, style, type hints, docstrings, config management, and packaging.

---

## Directory Layout

```
project/
├── checkpoints/          # Saved model weights (.pt), organized by dataset
├── configs/              # OmegaConf YAML files, one per experiment
├── data/                 # Actual dataset files (raw, processed, splits, etc.)
├── src/
│   ├── datasets/         # Dataset classes and data loading functions
│   ├── models/
│   │   ├── base_model.py # BaseModel ABC
│   │   └── ...           # Concrete model files (one model family per file)
│   ├── train/
│   │   └── train.py      # Training entry point
│   └── utils/
│       └── tools.py      # EarlyStopping, val_loop, and shared helpers
├── scripts/              # One-off utility scripts
└── main.py               # Entry point: load config, build dataloaders, call train()
```

- `data/` holds the actual dataset files on disk and is never imported by Python code.
- `src/datasets/` holds Python files: `Dataset` subclasses, per-dataset `load_<name>()` factory functions, and a `load_dataset()` dispatcher.

---

## Model Classes

### BaseModel

Every project has a single `BaseModel` that all concrete models inherit from.
It extends both `nn.Module` and `ABC`.

```python
class BaseModel(nn.Module, ABC):
    @abstractmethod
    def get_output(self, *args, **kwargs) -> Tensor:
        # Pure forward pass — no loss, no metrics
        ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> TensorDict:
        # Must return a TensorDict with at least "loss" and "metrics"
        ...

    @abstractmethod
    @torch.inference_mode()
    def validate_batch(self, *args, **kwargs) -> TensorDict:
        # Runs under inference_mode; returns a flat TensorDict of metric tensors
        ...

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
```

### Concrete Models

- Inherit `BaseModel` (or an intermediate abstract class that also inherits `BaseModel`).
- The `"metrics"` TensorDict in `forward()` should contain detached scalars — not part of the compute graph.

**Example:**

```python
class MyModel(BaseModel):
    def get_output(self, x: Tensor) -> Tensor:
        return self.network(x)

    def forward(self, x: Tensor, target: Tensor) -> TensorDict:
        output = self.get_output(x)
        loss = self.loss_fn(output, target)
        return TensorDict({"loss": loss, "metrics": TensorDict({"loss": loss.detach()})})

    @torch.inference_mode()
    def validate_batch(self, x: Tensor, target: Tensor) -> TensorDict:
        output = self.get_output(x)
        return TensorDict({"loss": ..., "acc": ...})
```

When a model family has both an abstract base and concrete variants (e.g., `VisionTransformer` → `ViTClassifier`), keep them in the same file.

---

## Dataset Classes

- Subclass `torch.utils.data.Dataset` (or an existing dataset like `torchvision.datasets.CIFAR10`).
- `__getitem__` returns a `TensorDict` with named keys (e.g., `"image"`, `"target"`) rather than a bare tuple. This makes the downstream training code self-documenting.
- Each dataset has a standalone `load_<name>()` factory function that:
  - Accepts a `DictConfig`
  - Handles train/val split (using `random_split` with a seeded generator)
  - Returns `(train_set, val_set, test_set)`.
- A `load_dataset(config: DictConfig)` dispatcher routes to the right factory based on `config.name`:

```python
def load_dataset(config: DictConfig) -> tuple[Dataset, Dataset, Dataset]:
    match config.name:
        case "cifar10":
            return load_cifar10(config)
        case _:
            raise ValueError(f"Unknown dataset: {config.name}")
```

---

## Training Loop

### `train_epoch()`

Runs one epoch of training and returns a merged `TensorDict` of train + val metrics.

```
train_epoch(model, train_loader, val_loader, optimizer, epoch, check_unit)
  -> TensorDict   # keys prefixed "train/" and "val/"
```

- Sets `model.train()` at the start.
- Calls `optimizer.zero_grad()` → `model(**batch)` → `output["loss"].backward()` → `optimizer.step()`.
- Accumulates per-step metrics in a `TensorDict`; every `check_unit` steps, prints the running mean of train metrics so far.
- `check_unit` controls the intermediate print frequency and is derived from `config.check_prop` (a fraction of epoch length).
- Calls `val_loop()` at the end of the epoch to get validation metrics, then prints the final combined metrics.

### `train()`

Outer loop that drives epochs, early stopping, and checkpointing.

```
train(model, config, train_loader, val_loader)
```

- Calls `torch.compile(model, ...)` when `config.compile` is `True`.
- Instantiates optimizer and `EarlyStopping` from config fields.
- Calls `train_epoch()` each epoch.
- On `EarlyStopping.check()`: breaks if `early_stop`, saves checkpoint if `improved`.

---

## Validation

`val_loop()` lives in `src/utils/tools.py` and is shared across training scripts.

```python
@torch.inference_mode()
def val_loop(model: BaseModel, val_loader: DataLoader, kind: str = "val") -> TensorDict:
    ...
    # Returns TensorDict with keys "{kind}/{metric}"
```

- Sets `model.eval()`.
- Calls `model.validate_batch(**batch)` for each batch.
- Aggregates over batches using `TensorDict`, then returns mean metrics prefixed with `kind/`.

---

## Early Stopping

`EarlyStopping` lives in `src/utils/tools.py`.

```python
class EarlyStopping:
    def __init__(self, val_key: str, tolerance: int, higher_better: bool): ...
    def check(self, metrics: TensorDict) -> tuple[bool, bool]:
        # Returns (early_stop, improved)
        ...
```

- `val_key` references a key in the metrics `TensorDict` (e.g., `"val/acc"`).
- `tolerance` is the number of non-improving epochs before stopping.
- `higher_better` controls whether improvement means going up or down.
- Returns `(early_stop, improved)` — use `improved` to decide whether to save a checkpoint.

---

## Configuration

Config files live in `configs/` as YAML, loaded with `OmegaConf.load()` in `main.py`.

**Typical top-level keys:**

```yaml
epochs: 100
compile: true
gpu_num: 0
model_save_dir: checkpoints/${data_name}
model_save_path: ${model_save_dir}/${model_name}.pt
check_prop: 0.01  # fraction of epoch length between intermediate metric prints

dataset:
  name: cifar10
  root: data/

dataloader:
  train:
    batch_size: 128
    shuffle: true
    pin_memory: true
  val:
    batch_size: 512
    shuffle: false

optimizer:
  lr: 1e-4

early_stopping:
  val_key: val/acc
  tolerance: 5
  higher_better: true

model:
  num_classes: 10
```

Pass the full `DictConfig` (or a sub-config) down into functions rather than extracting individual values at the top level.

---

## Entry Point (`main.py`)

`main.py` is thin: load config → build data → build model → call `train()`.

```python
def main():
    config = OmegaConf.load("configs/my_experiment.yaml")
    Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)

    train_set, val_set, _ = load_dataset(config.dataset)
    # collate_fn=torch.stack is required: TensorDict items must be stacked, default collate does not work
    train_loader = DataLoader(train_set, **config.dataloader.train, collate_fn=torch.stack)
    val_loader   = DataLoader(val_set,   **config.dataloader.val,   collate_fn=torch.stack)

    model = MyModel(**config.model).to(config.gpu_num)
    train(model, config, train_loader, val_loader)
```

---

## Key Libraries

| Purpose | Library |
|---|---|
| Neural networks | `torch`, `torch.nn` |
| Batch containers | `tensordict.TensorDict` |
| Config loading | `omegaconf.OmegaConf`, `DictConfig` |
| Data transforms | `torchvision.transforms` |
