#!/bin/bash
conda env create -f environment.yml
conda activate message-passing-nn
#export PYTHONPATH=path/to/message-passing-nn/
. grid-search-parameters.sh
python src/cli.py grid-search