#!/bin/bash

python setup.py sdist

docker build . -t message-passing-nn
docker container create --name message-passing-nn-container message-passing-nn:latest
echo "Removing untagged images"
# shellcheck disable=SC2046
docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
docker start message-passing-nn-container

docker logs -f message-passing-nn-container