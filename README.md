[![Build Status](https://dev.azure.com/kovamos/message-passing-nn/_apis/build/status/kovanostra.message-passing-nn?branchName=master)](https://dev.azure.com/kovamos/message-passing-nn/_build/latest?definitionId=2&branchName=master)
### Table of contents
- [1. Description](#1-description)
- [2. Requirements](#2-requirements)
- [3. Environment](#3-environment)
- [4. Dataset](#4-dataset)
- [5. Environment variables](#5-environment-variables)
- [6. Execute a grid search](#6-execute-a-grid-search)
- [7. Import and use](#7-import-and-use)
- [8. Tox build](#8-tox-build)
- [9. Run the code using docker](#9-run-the-code-using-docker)
- [10. Azure pipelines project](#10-azure-pipelines-project)


### 1. Description

This repository contains:
1. A pytorch implementation of a message passing neural network with GRU units (inspired from https://arxiv.org/abs/1812.01070). 
2. A wrapper around the model to perform a grid search, and save model checkpoints when the validation error is best for each configuration.


### 2. Requirements
Python 3.7.6

Run
```
click
pytorch=1.4.0
pandas=1.03
```

Tests
```
numpy==1.17.4
pytorch=1.4.0
```

### 3. Environment

To create the "message-passing-nn" conda environment please run:
```
conda env create -f environment.yml
```

### 4. Dataset

This repository contains two dataset folders with examples of data to run the code:

    - sample-dataset (CPU compatible): Contains just one pair of features/labels with some default values. This dataset lets you run the code in demo mode.
    - protein-folding (Needs GPU): Contains pairs of features/labels for various proteins (prepared using https://github.com/simonholmes001/structure_prediction). The features represent protein characteristics, and the labels the distance between all aminoacids.

The repository expects the data to be in the following format:

    - filenames: something_features.pickle & something_labels.pickle
    - features: torch.tensor.Size(M,N)
    - labels: torch.tensor.Size(M,M)
    * All features and labels should be preprocessed to be of the same size
    
For example, in the protein-folding dataset:

    - M: represents the number of aminoacids
    - N: represents the number of protein features

### 5. Environment variables

The model and grid search can be set up using a set of environment variables contained in the grid-search-parameters.sh. 

**NOT USED FOR GRID SEARCH**

- Your dataset folder is defined by: 

DATASET_NAME='sample-dataset'

- Your dataset directory is defined by: 

DATA_DIRECTORY='data/'

- The directory to save the model checkpoints is defined by: 

MODEL_DIRECTORY='model'

- The directory to save the grid search results per configuration is defined by: 

RESULTS_DIRECTORY='results'

- The option to run the model on 'cpu' or 'cuda' can be controlled by: 

DEVICE='cpu'

**USED FOR GRID SEARCH**

To define a range for the grid search please pass the values in the following format:
1. For numeric ranges: ENVVAR='min_value&max_value&number_of_values' (e.g. '10&15&2')
2. For string ranges: ENVVAR='selection_1&selection_2' (e.g. 'SGD&Adam')

- The total number of epochs can be controlled by:

EPOCHS='10'

- The choice of the loss function can be controlled by (see src/fixtures/loss_functions.py for a full list):

LOSS_FUNCTION='MSE'

- The choice of the optimizer can be controlled by (see src/fixtures/optimizers.py for a full list):

OPTIMIZER='SGD'

- The batch size can be controlled by:

BATCH_SIZE='1'

- The validation split can be controlled by:

VALIDATION_SPLIT='0.2'

- The test split can be controlled by:

TEST_SPLIT='0.1'

- The message passing time steps can be controlled by:

TIME_STEPS='5'

- The number of epochs to evaluate the model on the validation set can be controlled by:

VALIDATION_PERIOD='5'

### 6. Execute a grid search

The grid search can be executed by executing a shell script:
```
. grid-search.sh
```

This script will:

1. Create the conda environment from the environment.yml (if not created already)
2. Activate it
3. If necessary export the PYTHONPATH=path/to/message-passing-nn/ (line needs to be uncommented first)
4. Export the environment variables to be used for the Grid Search
5. Run the grid search

### 7. Import and use
To install the project using pip please run:

```
pip install git+https://github.com/kovanostra/message-passing-nn/
```

The code can be used to either just train a configuration of the message passing neural network or to perform a whole grid search.

#### Train a configuration

To train one configuration of the model please execute the following (I use example values):
```
import torch
from src.domain.model_trainer import ModelTrainer
from src.domain.graph_encoder import GraphEncoder
from src.domain.graph import Graph
from src.domain.data_preprocessor import DataPreprocessor
from src.repository.training_data_repository import TrainingDataRepository

# Set up the variables 
device = 'cpu'
epochs = 10
loss_function = 'MSE'
optimizer = 'SGD'
batch_size = 1
validation_split = 0.2
test_split = 0.1
time_steps = 5
validation_period = 5

dataset_size = 10
number_of_nodes = 10
number_of_node_features = 2
fully_connected_layer_input_size = number_of_nodes*number_of_node_features
fully_connected_layer_output_size = number_of_nodes**2

# Set up the datasets. Please load your own dataset by uncommenting the following part.
# dataset_name = 'the-name-of-the-directory-containing-your-dataset'
# data_directory = 'the-path-to-the-directory-containing-all-your-datasets'
# training_data_repository = TrainingDataRepository(data_directory, dataset_name)
# raw_dataset = training_data_repository.get_all_features_and_labels_from_separate_files()
# initialization_graph = DataPreprocessor.extract_initialization_graph(raw_dataset)

# This is just an example to make the code runnable 
node_features_example = torch.ones(number_of_nodes, number_of_node_features) 
adjacency_matrix_example = torch.ones(number_of_nodes, number_of_nodes)
raw_dataset = [(node_features_example, adjacency_matrix_example) for i in range(dataset_size)]
training_data, validation_data, test_data = DataPreprocessor.train_validation_test_split(raw_dataset, 
                                                                                         batch_size, 
                                                                                         validation_split, 
                                                                                         test_split)
initialization_graph = Graph(adjacency_matrix_example, node_features_example)


graph_encoder = GraphEncoder(time_steps, 
                             number_of_nodes, 
                             number_of_node_features, 
                             fully_connected_layer_input_size, 
                             fully_connected_layer_output_size)
configuration_dictionary = {'time_steps': time_steps,
                            'loss_function': loss_function,
                            'optimizer': optimizer}
model_trainer = ModelTrainer(GraphEncoder, device)
model_trainer.instantiate_attributes(initialization_graph, configuration_dictionary)

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
from src.message_passing_nn import create

message_passing_nn = create(dataset_name='the-name-of-the-directory-containing-your-dataset',
                            data_directory='the-path-to-the-directory-containing-all-your-datasets',
                            model_directory='model',
                            results_directory='results',
                            device='cpu',
                            epochs='10&15&2',
                            loss_function_selection='MSE',
                            optimizer_selection='SGD',
                            batch_size='1',
                            validation_split='0.2&0.3&2',
                            test_split='0.1',
                            time_steps='2&5&2',
                            validation_period='5&15&3')
message_passing_nn.start()
```
In the above example please note that all values must be passed as strings.

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
The model can be run from inside a docker container. To do so please execute the following shell script:
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
