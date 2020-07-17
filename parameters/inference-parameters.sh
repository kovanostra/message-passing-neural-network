#!/usr/bin/env bash
export DATASET_NAME='direct_neighbour_full_labels_test'
export DATA_DIRECTORY='data/'
export MODEL_DIRECTORY='model_checkpoints/configuration&id__model&RNN__epochs&400__loss_function&MSE__optimizer&Adagrad__batch_size&100__maximum_number_of_features&-1__maximum_number_of_nodes&-1__validation_split&0.2__test_split&0.1__time_steps&1__validation_period&20/Epoch_400_model_state_dictionary.pth'
export RESULTS_DIRECTORY='results_inference'
export MODEL='RNN'
export DEVICE='cpu'
