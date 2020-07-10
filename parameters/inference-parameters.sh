#!/usr/bin/env bash
export DATASET_NAME='sample-dataset'
export DATA_DIRECTORY='data/'
#export MODEL_DIRECTORY='model_checkpoints/configuration&id__model&RNN__epochs&200__loss_function&MSE__optimizer&SGD__batch_size&15__maximum_number_of_features&-1__maximum_number_of_nodes&-1__validation_split&0.2__test_split&0.1__time_steps&2__validation_period&30/Epoch_90_model_state_dictionary.pth'
export MODEL_DIRECTORY='model_checkpoints/configuration&id__model&RNN__epochs&10000__loss_function&MSE__optimizer&SGD__batch_size&10__maximum_number_of_features&-1__maximum_number_of_nodes&-1__validation_split&0.2__test_split&0.1__time_steps&2__validation_period&30/Epoch_9990_model_state_dictionary.pth'
export RESULTS_DIRECTORY='grid_search_results'
export MODEL='RNN'
export DEVICE='cpu'
