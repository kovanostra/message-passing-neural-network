[![Build Status](https://dev.azure.com/kovamos/message-passing-nn/_apis/build/status/kovanostra.message-passing-nn?branchName=master)](https://dev.azure.com/kovamos/message-passing-nn/_build/latest?definitionId=2&branchName=master)
### Table of contents
- [1. Description](#1-description)
- [2. Import and use](#2-import-and-use-package)
- [3. Requirements](#3-requirements)
- [4. Environment](#4-environment)
- [5. Dataset](#5-dataset)
- [6. Environment variables](#6-environment-variables)
- [7. Execute a grid search](#7-execute-a-grid-search)
- [8. Tox build](#8-tox-build)
- [9. Run the code using docker](#9-run-the-code-using-docker)
- [10. Azure pipelines project](#10-azure-pipelines-project)


### 1. Description

This repository contains:
1. A pytorch implementation of a message passing neural network with either RNN or GRU units (inspired from https://arxiv.org/abs/1812.01070). 
2. A wrapper around the model to perform a grid search, and save model checkpoints when the validation error is best for each configuration.

Google colab notebook containing examples of using the code from this repository:
https://colab.research.google.com/drive/1jFJ7l7jIv22BhvvzlmXOWFtgBE15ea2X

### 2. Import and use package
To install the project using pip please run:

```
pip install message-passing-nn
```

The code can be used to either just train a configuration of the message passing neural network or to perform a whole grid search.

#### Train a configuration

To train one configuration of the model please execute the following (I use example values):
```
import torch
from message_passing_nn.trainer.model_trainer import ModelTrainer
from message_passing_nn.model.graph_gru_encoder import GraphGRUEncoder
from message_passing_nn.model.graph_rnn_encoder import GraphRNNEncoder
from message_passing_nn.data.data_preprocessor import DataPreprocessor

# Set up the variables 
device = 'cpu' # You can use 'cuda' for GraphRNNEncoder, but it is not adviced for GraphGRUEncoder
epochs = 10
loss_function = 'MSE'
optimizer = 'SGD'
batch_size = 1
maximum_number_of_features=-1
maximum_number_of_nodes=-1
validation_split = 0.2
test_split = 0.1
time_steps = 5
validation_period = 5

dataset_size = 10
number_of_nodes = 10
number_of_node_features = 2

# Set up the dataset. 

# To load your own dataset please uncomment the following part:
# from message_passing_nn.repository.file_system_repository import FileSystemRepository
# dataset_name = 'the-name-of-the-directory-containing-your-dataset'
# data_directory = 'the-path-to-the-directory-containing-all-your-datasets'
# file_system_repository = FileSystemRepository(data_directory, dataset_name)
# raw_dataset = file_system_repository.get_all_data()

# This is just an example to make the code runnable 
node_features_example = torch.tensor([[1, 2], 
                                     [1, 1], 
                                     [2, 0.5], 
                                     [0.5, 0.5]]).float()
adjacency_matrix_example = torch.tensor([[0, 1, 1, 0],
                                         [1, 0, 1, 0],
                                         [1, 1, 0, 1],
                                         [0, 0, 1, 0]]).float()
labels_example = torch.tensor([2, 0.5, 0.5, 0.5]).float()
raw_dataset = [(node_features_example, adjacency_matrix_example, labels_example) for i in range(dataset_size)]
data_preprocessor = DataPreprocessor()
equalized_dataset = data_preprocessor.equalize_dataset_dimensions(raw_dataset,
                                                                  maximum_number_of_nodes,
                                                                  maximum_number_of_features)
training_data, validation_data, test_data = data_preprocessor.train_validation_test_split(equalized_dataset, 
                                                                                          batch_size, 
                                                                                          validation_split, 
                                                                                          test_split)
data_dimensions = data_preprocessor.extract_data_dimensions(equalized_dataset)


configuration_dictionary = {'time_steps': time_steps,
                           'loss_function': loss_function,
                           'optimizer': optimizer}
#model_trainer = ModelTrainer(GraphGRUEncoder, data_preprocessor, device)
model_trainer = ModelTrainer(GraphRNNEncoder, data_preprocessor, device)
model_trainer.instantiate_attributes(data_dimensions, configuration_dictionary)

for epoch in range(epochs):
   training_loss = model_trainer.do_train(training_data, epoch)
   print("Epoch", epoch, "Training loss:", training_loss)
   if epoch % validation_period == 0:
       validation_loss = model_trainer.do_evaluate(validation_data, epoch)
       print("Epoch", epoch, "Validation loss:", validation_loss)
test_loss = model_trainer.do_evaluate(test_data)
print("Test loss:", validation_loss)
```

##### Perform a grid search
To perform a grid search please execute the following (I use example values for a grid search of 24 combinations):
```
from message_passing_nn.create_message_passing_nn import create

message_passing_nn = create(dataset_name='the-name-of-the-directory-containing-your-data',  #e.g. 'sample-dataset/'
                           data_directory='the-path-to-the-directory-containing-all-your-datasets', #e.g. '~/message-passing-nn/data/'
                           model_directory='model_checkpoints',
                           results_directory='grid_search_results',
                           model='RNN',
                           device='cpu',
                           epochs='10&15&2',
                           loss_function_selection='MSE',
                           optimizer_selection='SGD',
                           batch_size='1',
                           maximum_number_of_features='-1',
                           maximum_number_of_nodes='-1',
                           validation_split='0.2&0.3&2',
                           test_split='0.1',
                           time_steps='2&5&2',
                           validation_period='5&15&3')
message_passing_nn.start()
```
In the above example please note that all values must be passed as strings.

### 3. Requirements

Python 3.7.6

Run
```
click
pytorch=1.4.0
numpy==1.17.4
pandas=1.03
```

Tests
```
numpy==1.17.4
pytorch=1.4.0
```

### 4. Environment
To create the "message-passing-nn" conda environment please run:

```
conda env create -f environment.yml
```

### 5. Dataset
      
This repository contains two dataset folders with examples of data to run the code:

  - sample-dataset: Contains just one pair of features/labels with some default values. This data lets you run the code in demo mode.
  - protein-folding: Contains pairs of features/labels for various proteins (prepared using https://github.com/simonholmes001/structure_prediction). The features represent protein characteristics, and the labels the distance between all aminoacids.

The repository expects the data to be in the following format:

  - filenames: something_features.pickle, something_adjacency-matrix.pickle & something_labels.pickle
  - features: torch.tensor.Size([M,N])
  - adjacency-matrix: torch.tensor.Size([M,M])
  - labels: torch.tensor.Size([L])
  * All features and labels should be preprocessed to be of the same size
  
For example, in the protein-folding dataset:

  - M: represents the number of amino acids
  - N: represents the number of protein features
  - L: represents the number of values to predict

### 6. Environment variables
      
The model and grid search can be set up using a set of environment variables contained in the grid-search-parameters.sh. 

**NOT USED FOR GRID SEARCH**

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

- The choice of the loss function can be controlled by (see src/fixtures/loss_functions.py for a full list):

LOSS_FUNCTION='MSE'

- The choice of the optimizer can be controlled by (see src/fixtures/optimizers.py for a full list):

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

### 7. Execute a grid search
       
Before executing a grid-search please go to the grid-search.sh to add your PYTHONPATH=path/to/message-passing-nn/.

The grid search can be executed by executing a shell script:
```
. grid-search.sh
```

This script will:

1. Create the conda environment from the environment.yml (if not created already)
2. Activate it
3. It exports the PYTHONPATH=path/to/message-passing-nn/ (line needs to be uncommented first)
4. Export the environment variables to be used for the Grid Search
5. Run the grid search

### 8. Tox build

Tox is a tool which downloads the code dependencies, runs all the tests and, if the tests pass, it builds an artifact in the .tox/dist/ directory. The artifact is name tagged by the version of your code which can be specified in the setup.py.

Requirements

```
click
tox==3.14.3
pytorch=1.4.0
numpy==1.17.4
pandas=1.03
```

From the parent directory run (with sudo if necessary):
```
tox
```

### 9. Run the code using docker
The model can be run from inside a docker container (currently cpu only). To do so please execute the following shell script:
```
. grid-search-docker.sh
```

The grid-search-docker.sh will:

    1. Remove any previous message-passing-nn containers and images
    2. Build the project
    3. Create a docker image
    4. Create a docker container
    5. Start the container
    6. Print the containner's logs with the --follow option activated

By default the dockerfile uses the sample-dataset. To change that please modify the grid-search-parameters.sh.

You can clear the docker container and images created by running:
```
. remove-containers-and-images.sh
```
This, by default will remove only tagged images created by the train-model.sh. However, you can uncomment the following lines if you want to remove all stopped containers and untagged images too:
```
docker container rm $(docker container ls -aq)
docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
```
Please note that this will delete also untagged images created by other projects, so use with caution.

### 10. Azure pipelines project

https://dev.azure.com/kovamos/message-passing-nn
