#!/bin/bash
# Linux version 4.4.0-130-generic (buildd@lgw01-amd64-039) (gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.9) )
echo "Keras + TensorFlow GPU Installer Part 2 .::. Version 1.0"
echo "More info at: https://www.pyimagesearch.com/2017/09/27/setting-up-ubuntu-16-04-cuda-gpu-for-deep-learning-with-python/"

# checks if cuda_9.2.148_396.37_linux-run exists, if so, install...
if [ ! -f cuda_9.0.176_384.81_linux-run ]; then
	echo "********** DOWNLOADING cuda_9.0.176_384.81_linux-run **********"
    sudo wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
fi

echo "1/3 ********** INSTALLING CUDA TOOLKIT cuda_9.0.176_384.81_linux-run **********"
sudo bash -c 'chmod +x cuda_9.0.176_384.81_linux-run'
mkdir installers
sudo ./cuda_9.0.176_384.81_linux-run -extract=`pwd`/installers
cd installers
sudo ./NVIDIA-Linux-x86_64-384.81.run
modprobe nvidia
sudo ./cuda-linux.9.0.176-22781540.run
sudo ./cuda-samples.9.0.176-22781540-linux.run
cd /tmp

# echo "2/3 ********** UPDATING ~/.bashrc **********"
sudo echo "" >> ~/.bashrc
sudo echo "# NVIDIA CUDA Toolkit" >> ~/.bashrc
sudo echo "export PATH=/usr/local/cuda-9.0/bin:$PATH" >> ~/.bashrc
sudo echo "export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/" >> ~/.bashrc
source ~/.bashrc

# echo "3/3 ********** TESTING CUDA TOOLKIT INSTALLATION **********"
cd /usr/local/cuda-9.0/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
cd /tmp

echo ""
echo "######################################################"
echo "################## !!! ATTENTION !!! #################"
echo "########## MAKE SURE YOU GOT: Result = PASS ##########"
echo "######################################################"
echo "################## !!! ATTENTION !!! #################"
echo ""
echo "Download CuDNN library at https://developer.nvidia.com/cudnn cuDNN and run part 3 of this script."