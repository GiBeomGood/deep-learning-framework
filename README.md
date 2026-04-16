# Deep Learning Study & Implementations

A professional repository dedicated to the study and implementation of various deep learning architectures, research papers, and experimental techniques. This repository serves as a personal knowledge base for deep learning concepts, featuring modular and extensible codebases.

## Introduction

This project aims to provide high-quality implementations of foundational and state-of-the-art deep learning models from scratch using **PyTorch**. Each model is implemented with a focus on readability and reproducibility, following common software engineering best practices for deep learning research.

The repository currently includes implementations of:

- **ResNet (Residual Networks)**: Deep residual learning for image recognition.
- **ViT (Vision Transformers)**: An image is worth 16x16 words; transformers for image recognition at scale.
- **FFC (Fourier Feature Coding)**: Learning in the frequency domain for efficient feature extraction.
- **Transformer**: The foundational attention-based architecture for sequence modeling.

## Project Structure

```text
.
├── main.py                 # Main entry point for model training
├── configs/                # OmegaConf (YAML) configuration files
├── src/                    # Core library source code
│   ├── data/               # Data loading and preprocessing pipelines
│   ├── models/             # Model architecture implementations
│   ├── training/           # Training loops, DDP support, and loss functions
│   └── utils/              # General-purpose utility functions
├── data/                   # Dataset storage (e.g., CIFAR-10)
├── checkpoints/            # Model checkpoints and logs
├── notebooks/              # Jupyter notebooks for experimentation and visualization
├── manuals/                # Documentation and workflow instructions
├── scripts/                # Shell scripts for formatting and automation
└── README.md               # Project documentation
```

## Usage

### Environment Setup

This project uses **zsh** as the primary shell. To set up the environment, it is recommended to use a virtual environment or Conda.

```zsh
# Clone the repository
git clone <repository_url>
cd deep-learning-framework

# Install dependencies
uv sync --frozen
```

### Running Experiments

Experiments are driven by configuration files in the `configs/` directory. By default, `main.py` uses the ResNet50 configuration on CIFAR-10.

To start a training session:

```zsh
uv run python main.py
```

To use a different configuration, you can modify `main.py` or use OmegaConf's CLI overrides (if implemented):

```zsh
# Current implementation requires updating main.py or config files
vim configs/resnet50.yaml
```

## Future Work

- [ ] Implementation of Diffusion Models
- [ ] Support for additional datasets (ImageNet, COCO)
- [ ] Integration with Weights & Biases for experiment tracking
