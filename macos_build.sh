#!/bin/bash

export MACOSX_DEPLOYMENT_TARGET=10.11
export CC=clang
export CXX=clang++
python setup.py install
