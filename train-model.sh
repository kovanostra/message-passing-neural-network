#!/bin/bash

python setup.py sdist

echo "Removing previous containers and images for message-passing-nn"
docker rm -f message-passing-nn-container
docker rmi message-passing-nn

echo "Building a new docker image and container"
docker build . -t message-passing-nn
docker container create --name message-passing-nn-container message-passing-nn:latest

echo "Starting message-passing-nn-container"
docker start message-passing-nn-container

docker logs -f message-passing-nn-container