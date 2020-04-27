#!/bin/bash
conda activate message-passing-nn
export PYTHONPATH=~/message-passing-nn
. grid_search_parameters.sh
python src/cli.py grid-search