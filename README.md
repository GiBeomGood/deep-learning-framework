# Deep Learning Study & Implementations

This repository serves as a personal archive for my deep learning studies, containing implementations of various neural network architectures, experimental notebooks, and training scripts. It is designed to be a modular and extensible framework for exploring deep learning concepts.

## Introduction

The goal of this project is to implement and understand state-of-the-art deep learning models from scratch. It currently includes implementations of:
- **ResNet (Residual Networks)**
- **ViT (Vision Transformers)**
- **FFC (Fourier Feature Coding)**

The codebase is structured to separate data loading, model definitions, and training loops, making it easy to experiment with different configurations.

## Project Structure

```text
.
├── configs/                # Configuration files (YAML) for experiments
│   └── *.yaml
├── data/                   # Dataset storage (e.g., CIFAR-10)
├── manuals/                # Documentation and workflow guides
├── notebooks/              # Jupyter notebooks for analysis and experiments
│   └── *.ipynb
├── scripts                 # shell scripts
│   └── *.sh
├── src/                    # Source code
│   ├── data/               # Data loaders and preprocessing
│   │   └── *.py
│   ├── models/             # Model architectures
│   │   ├── base_model.py
│   │   └── *.py
│   ├── training/           # Training loops and utilities
│   │   ├── train.py
│   │   └── train_ddp.py
│   └── utils/              # Helper functions
│       └── tools.py
├── main.py                 # Entry point for training
├── pyproject.toml          # Project configuration
└── README.md               # Project documentation
```

## Usage

### Prerequisites

Ensure you have Python installed. You will need standard deep learning libraries such as PyTorch and OmegaConf.

```zsh
pip install torch torchvision omegaconf tqdm
```

### Running the Training

To start a training session (e.g., ResNet50 on CIFAR-10), run the `main.py` script. The configuration is loaded from `configs/resnet50.yaml` by default.

```zsh
python main.py
```

### Configuration

You can modify experiment parameters in the `configs/` directory. For example, to change hyperparameters for ResNet, edit `configs/resnet50.yaml`.
