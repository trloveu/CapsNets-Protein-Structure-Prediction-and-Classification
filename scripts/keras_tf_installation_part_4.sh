#!/bin/bash
# Linux version 4.4.0-130-generic (buildd@lgw01-amd64-039) (gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.9) )
echo "Keras + TensorFlow GPU Installer Part 3 .::. Version 1.0"
echo "More info at: https://www.pyimagesearch.com/2017/09/27/setting-up-ubuntu-16-04-cuda-gpu-for-deep-learning-with-python/"

echo "1/6 ********** CREATING CONDA VIRTUAL ENVIROMENT **********"
conda create -n volumetric_gpu python=3.6 -y
source activate volumetric_gpu

echo "2/6 ********** INSTALLING VIRTUAL ENVIRONMENT DEPENDENCIES **********"
pip install opencv-python scipy matplotlib pillow imutils h5py requests progressbar2 scikit-learn scikit-image wget numpy keras-vis cython tqdm pydot
conda install -c anaconda mpi4py -y

echo "3/6 ********** INSTALLING TENSORFLOW GPU **********"
pip install tensorflow-gpu

echo "4/6 ********** INSTALLING KERAS **********"
pip install keras

echo "5/6 ********** TESTING TENSORFLOW GPU **********"
python -c "import tensorflow"

echo "6/6 ********** TESTING KERAS **********"
python -c "import keras"

echo ""
echo "DONE!"
