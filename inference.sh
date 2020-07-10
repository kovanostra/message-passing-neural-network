#!/bin/bash
conda env create -f environment.yml
conda activate message-passing-neural-network
#export PYTHONPATH=path/to/message-passing-neural-network/
. parameters/inference-parameters.sh
python message_passing_nn/cli.py inference