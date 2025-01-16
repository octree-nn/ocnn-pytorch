# O-CNN

**[Documentation](https://ocnn-pytorch.readthedocs.io)**

[![Documentation Status](https://readthedocs.org/projects/ocnn-pytorch/badge/?version=latest)](https://ocnn-pytorch.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/ocnn)](https://pepy.tech/project/ocnn)
[![Downloads](https://static.pepy.tech/badge/ocnn/month)](https://pepy.tech/project/ocnn)
[![PyPI](https://img.shields.io/pypi/v/ocnn)](https://pypi.org/project/ocnn/)

This repository contains the **pure PyTorch**-based implementation of
[O-CNN](https://wang-ps.github.io/O-CNN.html). The code has been tested with
`Pytorch>=1.6.0`, and `Pytorch>=1.9.0` is preferred. The *original*
implementation of O-CNN is based on C++ and CUDA and can be found
[here](https://github.com/Microsoft/O-CNN), which has received
[![stars - O-CNN](https://img.shields.io/github/stars/microsoft/O-CNN?style=social)](https://github.com/microsoft/O-CNN) and
[![forks - O-CNN](https://img.shields.io/github/forks/microsoft/O-CNN?style=social)](https://github.com/microsoft/O-CNN).


O-CNN is an octree-based 3D convolutional neural network framework for 3D data.
O-CNN constrains the CNN storage and computation into non-empty sparse voxels
for efficiency and uses the `octree` data structure to organize and index these
sparse voxels. Currently, this type of 3D convolution is known as Sparse
Convolution in the research community.


The concept of Sparse Convolution in O-CNN is the same with
[SparseConvNet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf),
[MinkowskiNet](https://github.com/NVIDIA/MinkowskiEngine), and
[SpConv](https://github.com/traveller59/spconv).
The key difference is that our O-CNN uses `octrees` to index the sparse voxels,
while these works use `Hash Tables`. However, I believe that `octrees` may be
the right choice for Sparse Convolution. With `octrees`, I can implement the
Sparse Convolution with pure PyTorch. More importantly, with `octrees`, I can
also build efficient transformers for 3D data --
[OctFormer](https://github.com/octree-nn/octformer), which is extremely hard
with `Hash Tables`.


Our O-CNN is published in SIGGRAPH 2017, SparseConvNet is published in CVPR
2018, and MinkowskiNet is published in CVPR 2019. Actually, our O-CNN was
submitted to SIGGRAPH in the end of 2016 and was officially accepted in March,
2017. <!-- The camera-ready version of our O-CNN was submitted to SIGGRAPH in April, 2018. -->
We just did not post our paper on Arxiv during the review process of SIGGRAPH.
Therefore, **the idea of constraining CNN computation into sparse non-emtpry
voxels, i.e. Sparse Convolution,  is first proposed by our O-CNN**.


Developed in collaboration with authors from [PointCNN](https://arxiv.org/abs/1801.07791),
[Dr. Yangyan Li](https://yangyan.li/) and [Prof. Baoquan Chen](https://baoquanchen.info/),
this library supports point cloud processing from the ground up.
The library provides essential components for converting raw point clouds into
octrees to perform convolution operations. Of course, it also supports other 3D
data formats, such as meshes and volumetric grids, which can be converted into
octrees to leverage the library's capabilities.


## Key benefits of ocnn-pytorch

- **Simplicity**. The ocnn-pytorch is based on pure PyTorch, it is portable and
  can be installed with a simple command:`pip install ocnn`. Other sparse
  convolution frameworks heavily rely on C++ and CUDA, and it is complicated to
  configure the compiling environment.

- **Efficiency**. The ocnn-pytorch is very efficient compared with other sparse
  convolution frameworks.  It only takes 18 hours to train the network on
  ScanNet for 600 epochs with 4 V100 GPUs. For reference, under the same
  training settings, MinkowskiNet 0.4.3 takes 60 hours and MinkowskiNet 0.5.4
  takes 30 hours.

## Citation

  ```bibtex
  @article {Wang-2017-ocnn,
    title    = {{O-CNN}: Octree-based Convolutional Neural Networksfor {3D} Shape Analysis},
    author   = {Wang, Peng-Shuai and Liu, Yang and Guo, Yu-Xiao and Sun, Chun-Yu and Tong, Xin},
    journal  = {ACM Transactions on Graphics (SIGGRAPH)},
    volume   = {36},
    number   = {4},
    year     = {2017},
  }
  ```
