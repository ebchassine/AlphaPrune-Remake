#!/bin/bash

# Apple Silicon Optimized Install Script for TempBalance (language_modeling)
# Creates a conda environment and installs dependencies safely on M1/M2 Macs

# Step 1: Define environment name
ENV_NAME="ww_train_lm"

# Step 2: Create environment with safe Python version
conda create -n $ENV_NAME python=3.8.16 -y

# Step 3: Activate environment (ensure this works in non-interactive shells)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Step 4: Install core dependencies via conda-forge/pytorch channels
conda install -y \
  numpy=1.22.2 \
  scipy=1.8.0 \
  pandas=1.4.0 \
  matplotlib=3.5.1 \
  scikit-learn=1.0.2 \
  pytorch=1.12.0 \
  torchvision=0.13.0 \
  pyyaml=6.0 \
  -c pytorch -c conda-forge

# Step 5: Install lightweight packages via pip
pip install adamp tabulate==0.8.10 termcolor==1.1.0 sklearn==0.0

echo "Environment '$ENV_NAME' is ready for use."

