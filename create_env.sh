#!/bin/bash

# create conda env
# conda create -n mega python=3.9

# Install PyTorch
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

read -r -p "Pytorch Installation Finish. Continue? [y/N] " response
case "$response" in
    [yY][eE][sS]|[yY]) 
        ;;
    *)
        exit 0
        ;;
esac

# Install Pytorch3d
# refer to https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub

conda install -y pytorch3d -c pytorch3d

read -r -p "Pytorch3d Installation Finish. Continue? [y/N] " response
case "$response" in
    [yY][eE][sS]|[yY]) 
        ;;
    *)
        exit 0
        ;;
esac

# Install other python packages
pip install opencv-python lpips kornia tensorboard sparse trimesh roma chumpy ninja
pip install submodules/diff-gauss submodules/nvdiffrast submodules/simple-knn

pip install numpy==1.23.1   # downgrade the numpy package