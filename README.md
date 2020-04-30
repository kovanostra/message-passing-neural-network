[![Build Status](https://dev.azure.com/kovamos/message-passing-nn/_apis/build/status/kovanostra.message-passing-nn?branchName=master)](https://dev.azure.com/kovamos/message-passing-nn/_build/latest?definitionId=2&branchName=master)

### Description

This repository contains a pytorch implementation of a message passing neural network with GRU units. It is inspired from Jin et al. ICLR 2019 (https://arxiv.org/abs/1812.01070).


### Requirements
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

### Environment

To create the "message-passing-nn" conda environment please run:
```
conda env create -f environment.yml
```

### Dataset

This repository contains two dataset folders with examples of data to run the code:

    - sample-dataset (CPU compatible): Contains just one pair of features/labels with some default values. This dataset lets you run the code in demo mode.
    - protein-folding (Needs GPU): Contains pairs of features/labels for various proteins (prepared using https://github.com/simonholmes001/structure_prediction). The features represent protein characteristics, and the labels the distance between all aminoacids.

The repository expects the data to be in the following format:

    - filenames: something_features.pickle & something_labels.pickle
    - features: torch.tensor.Size(M,N)
    - labels: torch.tensor.Size(M,M)
    - All features and labels should be preprocessed to be of the same size
    
For example, in the protein-folding dataset:

    - M: represents the number of aminoacids
    - N: represents the number of protein features

### Run the code

The code can be run by executing a shell script:
```
. grid-search.sh
```

This script will

1. Create the conda environment from the environment.yml (if not created already)
2. Activate it
3. If necessary export the PYTHONPATH=path/to/message-passing-nn/ (line needs to be uncommented first)
4. Export the environment variables to be used for the Grid Search
5. Run the grid search

### Tox build

Tox is a tool which downloads the code dependencies, runs all the tests and, if the tests pass, it builds an artifact in the .tox/dist/ directory. The artifact is name tagged by the version of your code which can be specified in the setup.py.

Requirements

```
click
tox==3.14.3
pytorch=1.4.0
numpy==1.17.4
pandas=1.03
```

From the message-passing-nn/ directory run (with sudo if necessary):
```
tox
```
### Run the code using docker
The model can be run from inside a docker container. To do so please execute the following shell script from inside ~/message-passing-nn/:
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

You can clear the docker container and images created by running again from inside ~/message-passing-nn/:
```
. remove-containers-and-images.sh
```
This, by default will remove only tagged images created by the train-model.sh. However, you can uncomment the following lines if you want to remove all stopped containers and untagged images too:
```
docker container rm $(docker container ls -aq)
docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
```
Please note that this will delete also untagged images created by other projects, so use with caution.

### Azure pipelines project

https://dev.azure.com/kovamos/message-passing-nn
