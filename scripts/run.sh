#!/usr/bin/zsh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch
python main.py
