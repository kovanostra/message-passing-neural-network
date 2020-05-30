[![Build Status](https://dev.azure.com/kovamos/message-passing-nn/_apis/build/status/kovanostra.message-passing-nn?branchName=master)](https://dev.azure.com/kovamos/message-passing-nn/_build/latest?definitionId=2&branchName=master)
### Table of contents
- [1. Description](#1-description)
- [2. Import and use](#2-import-as-package)
- [3. Examples](#3-examples)
- [4. Requirements](#4-requirements)
- [5. Environment](#5-environment)
- [6. Dataset](#6-dataset)
- [7. Environment variables](#7-environment-variables)
- [8. Execute a grid search](#8-execute-a-grid-search)
- [9. Tox build](#9-tox-build)
- [10. Run the code using docker](#10-run-the-code-using-docker)
- [11. Azure pipelines](#11-azure-pipelines)
- [12. Upcoming](#12-upcoming)


### 1. Description

This repository contains:
1. A pytorch implementation of a message passing neural network with either RNN or GRU units (inspired from https://arxiv.org/abs/1812.01070). 
2. A wrapper around the model to perform a grid search, and save model checkpoints when the validation error is best for each configuration.

### 2. Import as package
To install the project using pip please run:

```
pip install message-passing-nn
```

### 3. Examples

The code can be used to either train a single configuration of the message passing neural network or to perform a grid search. For usage examples please look in the example_notebooks/ directory or on the [colab notebook](https://colab.research.google.com/drive/1jFJ7l7jIv22BhvvzlmXOWFtgBE15ea2X).

### 4. Requirements

Python 3.7.6

Run
```
click
torch=1.4.0
numpy==1.17.4
pandas=1.0.3
```

Tests
```
numpy==1.17.4
torch=1.4.0
pandas=1.0.3
```

### 5. Environment
To create the "message-passing-nn" conda environment please run:

```
conda env create -f environment.yml
```

### 6. Dataset
      
The repository expects the data to be in the following format:

  - filenames: something_features.pickle, something_adjacency-matrix.pickle & something_labels.pickle
  - features: torch.tensor.Size([M,N])
  - adjacency-matrix: torch.tensor.Size([M,M])
  - labels: torch.tensor.Size([L])

**All features and labels should be preprocessed to be of the same size**
  
For example, in the protein-folding dataset:

  - M: represents the number of amino acids
  - N: represents the number of protein features
  - L: represents the number of values to predict

This repository contains two dataset folders with examples of data to run the code:

  - sample-dataset: Contains just one pair of features/labels with some default values. This data lets you run the code in demo mode.
  - protein-folding: Contains pairs of features/labels for various proteins (prepared using https://github.com/simonholmes001/structure_prediction). The features represent protein characteristics, and the labels the distance between all aminoacids.

### 7. Environment variables
      
The model and grid search can be set up using a set of environment variables contained in the grid-search-parameters.sh. Please refer to the ENVIRONMENT_VARIABLES.md for the full list of available environment variables and how to use them.

### 8. Execute a grid search
       
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

### 9. Tox build

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

### 10. Run the code using docker
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

### 11. Azure pipelines

https://dev.azure.com/kovamos/message-passing-nn

### 12. Upcoming

V1.5.0:

      - The RNN model will be implemented in C++ (both forward and backward methods)
