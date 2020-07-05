#!/bin/bash
conda env create -f environment.yml
conda activate message-passing-nn
#FOR MAC
export MACOSX_DEPLOYMENT_TARGET=10.11
export CC=clang
export CXX=clang++
#FOR LINUX
#export CC=gcc
#export CXX=g++
python setup.py install
#export PYTHONPATH=path/to/message-passing-neural-network/conda remove --name myenv --all
. inference.sh
python message_passing_nn/cli.py inference