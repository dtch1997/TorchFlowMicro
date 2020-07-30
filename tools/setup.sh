#!/bin/bash

# Make sure cuda toolkit corresponds to what is installed on system
# Check cuda toolkit version using 'nvcc --version'
ENV_NAME="pipeline"
CUDATOOLKIT_VERSION="10.1"

git submodule update --init --recursive

conda create --name $ENV_NAME python=3.8
conda activate $ENV_NAME
conda install pytorch torchvision cudatoolkit=$CUDATOOLKIT_VERSION -c pytorch -y

python -m pip install -r requirements/onnx-tf.txt
cd lib/onnx-tensorflow
python -m pip install -e .



