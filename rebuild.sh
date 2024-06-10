#!/bin/bash

rm -rf build
mkdir build
cd build
cmake .. -Wno-dev
make -j4 
cd ..