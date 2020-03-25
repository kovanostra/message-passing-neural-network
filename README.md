[![Build Status](https://dev.azure.com/kovamos/message-passing-nn/_apis/build/status/kovanostra.protein-folding?branchName=master)](https://dev.azure.com/kovamos/message-passing-nn/_build/latest?definitionId=2&branchName=master)

### Description

This repository contains a pytorch implementation of a message passing neural network with GRU units. It is inspired from Jin et al. ICLR 2019 (https://arxiv.org/abs/1812.01070).


### Requirements
Python 3.7.6

Run
```
click
numpy==1.17.4
pytorch=1.4.0
```

Build
```
click
tox==3.14.3
numpy==1.17.4
pytorch=1.4.0
```

To run all tests and build the project, just cd to ~/message-passing-nn/ and run (with sudo if necessary)
```
tox
```

This will build an artifact and place it in ~/message-passing-nn/.tox/dist/graph-to-graph-version.zip. The version can be specified in the setup.py. The contents of this folder are cleaned at the start of every new build.

### Entrypoint

To start training the model please run the following from inside ~/message-passing-nn/:
```
python src/ci.py start-training
```

The model runs with default values for the number of epochs (10), loss function ('MSE') and optimizer ('SGD'). However, these can be changed as seen below:
 ```
 python src/ci.py start-training --epochs 10 -- loss_function 'SGD' --optimizer 'adam'
 ```

### Docker
The model can be run from inside a docker container. To do so please execute the following shell script from inside ~/message-passing-nn/:
```
. train-model.sh
```
The train-model.sh will:

    - Build the project
    - Create a docker image
    - Create a docker container
    - Start the container
    - Print the containner's logs with the --follow option activated

Afterwards, you can clear the docker container and images created by running again from inside ~/message-passing-nn/:
```
. remove-containers-and-images.sh
```
This, by default will not remove any untagged images created by the train-model.sh. However, you can uncomment the following line if you want to do so:
```
docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
```
Please note that this will delete also untagged images created by other projects, so use with caution.

### Azure pipelines project

https://dev.azure.com/kovamos/message-passing-nn
