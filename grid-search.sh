#!/bin/bash
conda env create -f environment.yml
conda activate message-passing-nn
export MACOSX_DEPLOYMENT_TARGET=10.11
export CC=clang
export CXX=clang++
python setup.py install
#export PYTHONPATH=path/to/message-passing-nn/
. grid-search-parameters.sh
python message_passing_nn/cli.py grid-search