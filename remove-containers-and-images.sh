docker rm -f message-passing-nn-container
docker rmi message-passing-nn
# echo "Removing stopped containers and untagged images"
#docker container rm $(docker container ls -aq)
#docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
