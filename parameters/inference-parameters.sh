#!/usr/bin/env bash
export DATASET_NAME='direct_neighbour_full_labels_test'
export DATA_DIRECTORY='data/'
export MODEL_DIRECTORY='model_checkpoints/configuration&id__model&RNN__epochs&1000__loss_function&MSE__optimizer&Adagrad__batch_size&200__validation_split&0.2__test_split&0.1__time_steps&1__validation_period&5/100_nodes_best_validation.pth'
export RESULTS_DIRECTORY='results_inference'
export MODEL='RNN'
export DEVICE='cpu'
