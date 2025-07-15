#!/bin/bash
set -e

echo "Setting up CUDA Math Labs development environment..."

# Install Python dependencies
pip install -r requirements.txt

# Setup pre-commit hooks
pre-commit install

# Build all submodules
git submodule update --init --recursive

echo "Development environment ready!"
