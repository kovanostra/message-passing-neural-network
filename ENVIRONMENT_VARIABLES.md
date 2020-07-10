**GENERAL PARAMETERS**

- Your dataset folder is defined by: 

DATASET_NAME='sample-dataset'

- Your dataset directory is defined by: 

DATA_DIRECTORY='data/'

- The directory to save the model checkpoints is defined by: 

MODEL_DIRECTORY='model_checkpoints'

- The directory to save the grid search results per configuration is defined by: 

RESULTS_DIRECTORY='grid_search_results'

- The option to run the model on 'cpu' or 'cuda' can be controlled by (*cuda recommended only for the 'RNN' model*): 

DEVICE='cpu'

**USED FOR GRID SEARCH**

To define a range for the grid search please pass the values in the following format:
1. For numeric ranges: ENVVAR='min_value&max_value&number_of_values' (e.g. '10&15&2')
2. For string ranges: ENVVAR='selection_1&selection_2' (e.g. 'SGD&Adam')

- The model to use ('RNN' or 'GRU') is defined by :

MODEL='RNN'

- The total number of epochs can be controlled by:

EPOCHS='10'

- The choice of the loss function can be controlled by (see message_passing_nn/utils/loss_functions.py for a full list):

LOSS_FUNCTION='MSE'

- The choice of the optimizer can be controlled by (see message_passing_nn/utils/optimizers.py for a full list):

OPTIMIZER='SGD'

- The batch size can be controlled by:

BATCH_SIZE='1'

- If the number of features in your dataset is too large please change the following value (-1 will use all of them)

MAXIMUM_NUMBER_OF_FEATURES='-1'

- If the number of nodes in your dataset is too large please change the following value (-1 will use all of them)

MAXIMUM_NUMBER_OF_NODES='-1'

- The validation split can be controlled by:

VALIDATION_SPLIT='0.2'

- The test split can be controlled by:

TEST_SPLIT='0.1'

- The message passing time steps can be controlled by:

TIME_STEPS='5'

- The number of epochs to evaluate the model on the validation set can be controlled by:

VALIDATION_PERIOD='5'

**USED FOR INFERENCE**

- The model to load ('RNN' or 'GRU') is defined by :

MODEL='RNN'