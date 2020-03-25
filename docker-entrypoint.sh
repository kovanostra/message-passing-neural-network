#!/bin/bash

if [ "$1" = "message-passing-nn" ]; then
  if [ -f SUCCESS ]; then
    echo "Removing success file"
    rm SUCCESS
  fi
fi

exec "$@"