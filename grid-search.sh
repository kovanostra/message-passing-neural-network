#!/bin/bash
conda env create -f environment.yml
conda activate message-passing-nn
#export PYTHONPATH=path/to/message-passing-nn/
. grid_search_parameters.sh
python src/cli.py grid-search