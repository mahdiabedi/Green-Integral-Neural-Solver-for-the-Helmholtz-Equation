#!/bin/bash
# 
# Installer for the Green-Integral Neural Solver environment
# 
# Run: ./install_env.sh
# 

echo 'Creating Conda environment from environment.yml...'

# create conda env
conda env create -f environment.yml

# Source conda to allow activation within the bash script
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment 
conda activate GPU4
echo 'Activated environment. Python path:' $(which python)

# Install the local gi_solver package in developer mode
echo 'Installing gi_solver package...'
pip install -e .

# Check that TensorFlow is installed and can see the GPU
echo 'Checking TensorFlow version and GPU access...'
python -c '
import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs Available:", len(gpus))
if gpus:
    for gpu in gpus:
        print(" -", gpu.name)
print("Test Tensor successfully created:", tf.ones([1, 5]).numpy())
'

echo 'Done!'
