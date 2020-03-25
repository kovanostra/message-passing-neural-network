docker rm -f message-passing-nn-container
docker rmi message-passing-nn
# docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
