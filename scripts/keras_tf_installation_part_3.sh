#!/bin/bash
# Linux version 4.4.0-130-generic (buildd@lgw01-amd64-039) (gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.9) )
echo "Keras + TensorFlow GPU Installer Part 3 .::. Version 1.0"
echo "More info at: https://www.pyimagesearch.com/2017/09/27/setting-up-ubuntu-16-04-cuda-gpu-for-deep-learning-with-python/"

echo "1/2 ********** COPYING CUDNN TO THE LIB64 AND INCLUDE DIRS **********"
tar -zxf cudnn-9.0-linux-x64-v7.tgz
cd cuda
sudo cp -P lib64/* /usr/local/cuda/lib64/
sudo cp -P include/* /usr/local/cuda/include/
cd /tmp

echo "2/2 ********** INSTALLING MINICONDA **********"
# checks if cuda_9.2.148_396.37_linux-run exists, if so, install...
if [ ! -f Miniconda3-latest-Linux-x86_64.sh ]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi

sudo bash -c 'chmod +x Miniconda3-latest-Linux-x86_64.sh'
sudo ./Miniconda3-latest-Linux-x86_64.sh

echo ""
echo "######################################################"
echo "################## !!! ATTENTION !!! #################"
echo "####### CLOSE THE TERMINAL AND OPEN A NEW ONE ########"
echo "######################################################"
echo "################## !!! ATTENTION !!! #################"
echo ""
echo "Then, run part 4 of this script."