#!/usr/bin/env bash
export DATASET_NAME='sample-dataset'
export DATA_DIRECTORY='data/'
export MODEL_DIRECTORY='model_checkpoints'
export RESULTS_DIRECTORY='grid_search_results'
export MODEL='RNN'
export DEVICE='cpu'
export EPOCHS='10'
export LOSS_FUNCTION='MSE'
export OPTIMIZER='Adam'
export BATCH_SIZE='10'
export MAXIMUM_NUMBER_OF_FEATURES='-1'
export MAXIMUM_NUMBER_OF_NODES='30'
export VALIDATION_SPLIT='0.2'
export TEST_SPLIT='0.1'
export TIME_STEPS='1'
export VALIDATION_PERIOD='20'
