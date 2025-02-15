#!/bin/bash

rm -rf build
mkdir build
cd build
cmake .. -Wno-dev
make -j4 
cd ..
cp build/vss_vision/vss_vision.cpython-310-x86_64-linux-gnu.so vss_vision.so