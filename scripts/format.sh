#!/bin/zsh

uv run ruff format . --no-cache
uv run ruff check . --select I --fix --no-cache
