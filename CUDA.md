#!/bin/bash

### steps ####
# verify the system has a cuda-capable gpu
# download and install the nvidia cuda toolkit and cudnn
# setup environmental variables
# verify the installation
###

### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### If you have previous installation remove it first. 
sudo apt purge nvidia* -y
sudo apt remove nvidia-* -y
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt autoremove -y && sudo apt autoclean -y
sudo rm -rf /usr/local/cuda*

# system update
sudo apt update && sudo apt upgrade -y

# install other import packages
sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# first get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# find recommended driver versions for you
ubuntu-drivers devices

# install nvidia driver with dependencies
sudo apt install libnvidia-common-515 libnvidia-gl-515 nvidia-driver-515 -y

# reboot
sudo reboot now

# verify that the following command works
nvidia-smi

sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

# Update and upgrade
sudo apt update && sudo apt upgrade -y

 # installing CUDA-11.8
sudo apt install cuda-11-8 -y

# setup your paths
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

echo 'export PATH=/usr/local/cuda-11.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
# install cuDNN v11.8
# First register here: https://developer.nvidia.com/developer-program/signup

CUDNN_TAR_FILE="cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz"
sudo wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
sudo tar -xvf ${CUDNN_TAR_FILE}
sudo mv cudnn-linux-x86_64-8.7.0.84_cuda11-archive cuda

CUDNN_TAR_FILE="cudnn-linux-x86_64-8.0.5.39_cuda11-archive.tar.xz"
sudo wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/local_installers/11.0/cudnn-linux-x86_64-8.0.5.39_cuda11-archive.tar.xz

CUDNN_TAR_FILE="cudnn-11.0-linux-x64-v8.0.5.39.tgz"
sudo wget -O cudnn-11.0-linux-x64-v8.0.5.39.tgz https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.0_20201106/cudnn-11.0-linux-x64-v8.0.5.39.tgz?3DlvREUhtvqLbkzNJzlL1hHXeragxXjG5NXcbi97N5nJY9yDf2W8bdkXGFK5H9_O5LjaSp9QjzhRKI4AIkXuOqTHgZ9NKIKhpyvkoXD0pwqZFTuRA35dRxpztNZwsQ9UOTrAygci-VilDpGfmIeXDd47Dd4-1op7SZI5iIKayIx4t45Fgh_EPQV9AJVzg2bFzQgfZ-IcCs_0Zo8=&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9 
sudo tar -xvf ${CUDNN_TAR_FILE}
sudo mv cudnn-11.0-linux-x64-v8.0.5.39.tgz cuda

# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.8/include
sudo cp -P cuda/lib/libcudnn* /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*

sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.0/include
sudo cp -P cuda/lib/libcudnn* /usr/local/cuda-11.0/lib64/
sudo chmod a+r /usr/local/cuda-11.0/lib64/libcudnn*
# Finally, to verify the installation, check
nvidia-smi
nvcc -V




# EXTRA

1. Install CUDA 11.8
```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
$ sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
$ sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
$ sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install cuda
```

2. Install CUDNN 8.6.0 (Download from [here](https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz) first)
```
$ sudo apt-get install zlib1g
$ tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
$ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
$ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

3. Run the command below in Windows Command Prompt as admin (Thanks to [Roy Shilkrot](https://stackoverflow.com/questions/76016645/tensorflow-2-12-could-not-load-library-libcudnn-cnn-infer-so-8-in-wsl2)) and restart your WSL2
```
> cd \Windows\System32\lxss\lib
> del libcuda.so
> del libcuda.so.1
> mklink libcuda.so libcuda.so.1.1
> mklink libcuda.so.1 libcuda.so.1.1
```

4. Setup some WSL2 paths
```
$ echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
$ source ~/.bashrc
$ sudo ldconfig
```

5. Update some dependencies
```
$ sudo apt update
$ sudo apt upgrade
```

6. (Optional) Install Python 3.9.2 using pyenv

I used Python 3.9.2 for the exam, so I had to use pyenv in Ubuntu to not change Ubuntu's default Python. As for you, you can use Python 3.10.* and have no problem.
```
$ curl https://pyenv.run | bash
$ sudo apt install curl -y 
$ sudo apt install git -y
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo 'eval "$(pyenv init - --path)"' >> ~/.bashrc
$ exec $SHELL
$ sudo apt install build-essential curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev wget xz-utils zlib1g-dev
$ pyenv install 3.9.2
$ pyenv global 3.9.2
```

7. Install TensorFlow 2.13
```
$ pip install --upgrade pip
$ pip install tensorflow==2.13
```

8. Verify the GPU Setup
```
$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If a list of GPU devices is returned, you've installed TensorFlow successfully.