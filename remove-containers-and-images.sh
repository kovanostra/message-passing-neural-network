docker rm -f message-passing-nn-container
docker rmi message-passing-nn
# echo "Removing untagged images"
# docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
