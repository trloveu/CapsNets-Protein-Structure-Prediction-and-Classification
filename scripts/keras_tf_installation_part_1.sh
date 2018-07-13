#!/bin/bash
# Linux version 4.4.0-130-generic (buildd@lgw01-amd64-039) (gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.9) )
echo "Keras + TensorFlow GPU Installer Part 1 .::. Version 1.0"
echo "More info at: https://www.pyimagesearch.com/2017/09/27/setting-up-ubuntu-16-04-cuda-gpu-for-deep-learning-with-python/"
echo "1/7 ********** UPDATING OS **********"

# updates os
sudo apt-get -y update && sudo apt-get -y upgrade && sudo apt-get -y dist-upgrade

echo "2/7 ********** INSTALLING DEVELOPMENT TOOLS **********"
# installs development tools
sudo apt-get -y install build-essential cmake git unzip pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libhdf5-serial-dev graphviz libopenblas-dev libatlas-base-dev gfortran python-tk python3-tk python-imaging-tk

echo "3/7 ********** INSTALLING PYTHON DEVELOPMENT TOOLS **********"
# installs python development tools
sudo apt-get -y install python2.7-dev python3-dev

echo "4/7 ********** PREPARING FOR CUDA DRIVERS **********"
# prepares for cuda drivers
sudo apt-get -y install linux-image-generic linux-image-extra-virtual linux-source linux-headers-generic

echo "5/7 ********** DISABLING NOUVEAU KERNEL DRIVER **********"
# disables nouveau kernel driver
sudo touch /etc/modprobe.d/blacklist-nouveau.conf

sudo bash -c 'cat <<EOT > /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
EOT'

echo "6/7 ********** UPDATING THE INITIAL RAM FILESYSTEM **********"
# updates initial ram filesystem
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u

echo "7/7 ********** RESTARTING **********"
# restarts node
sudo reboot