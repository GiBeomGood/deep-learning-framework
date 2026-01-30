#!/bin/zsh

ruff format . --no-cache
ruff check . --select I --fix --no-cache
