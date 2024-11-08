#!/bin/bash

## Tested dockers:
# pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel 
# pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

## Usage:

# 1. Run the following command to enter the docker
: '
  docker run --gpus=all --rm -it --ipc=host --net=host                        \
  -v /home/ps/workspace:/home/ps/workspace -v /mnt:/mnt                       \
  --name=pytorch-1.11.0  pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel  
'

# 2. Run the following command in the folder containing `ocnn-pytorch`
: '
  cd /home/ps/workspace/docker
  bash ocnn2-pytorch/docker/bash.sh
'

# 3. Get the final output as follows:
: '
  ..................
  ----------------------------------------------------------------------
  Ran 18 tests in 4.709s

  OK
'
# 4. Commit and push the docker
: '
  docker ps -a
  docker commit <CONTAINER ID> wangps/ocnn:pytorch-1.11.0
  docker push wangps/ocnn:pytorch-1.11.0
'


## Start:

echo Copy the code
cp -r ocnn2-pytorch /workspace/ocnn-pytorch

echo Install tools
apt-get update
apt-get install -y --no-install-recommends git wget vim

echo Config bashrc
echo "export PS1='\[\033[1;36m\]\u\[\033[1;31m\]@(docker)\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\n\$\[\033[0m\] '" >> ~/.bashrc
source ~/.bashrc

echo Install ocnn-pytorch
cd /workspace/ocnn-pytorch
python setup.py install

echo Install packages
cd /workspace/ocnn-pytorch/projects
pip install -r requirements.txt

echo Run testing
cd /workspace/ocnn-pytorch
python -m unittest
