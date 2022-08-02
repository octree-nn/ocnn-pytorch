# O-CNN

**[Documentation](https://ocnn-pytorch.readthedocs.io)**

This repository contains the **pure PyTorch**-based implementation of
[O-CNN](https://wang-ps.github.io/O-CNN.html). The code has been tested with
`Pytorch>=1.9.0`.

O-CNN is an octree-based sparse convolutional neural network framework for 3D
deep learning. O-CNN constrains the CNN storage and computation into non-empty
sparse voxels for efficiency and uses the `octree` data structure to organize
and index these sparse voxels.

The concept of sparse convolution in O-CNN is the same with
[H-CNN](https://ieeexplore.ieee.org/abstract/document/8580422),
[SparseConvNet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf),
and
[MinkowskiNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf).
The key difference is that our O-CNN uses the `octree` to index the sparse
voxels, while these 3 works use the `Hash Table`.

Our O-CNN is published in SIGGRAPH 2017, H-CNN is published in TVCG 2018,
SparseConvNet is published in CVPR 2018, and MinkowskiNet is published in CVPR
2019. Actually, our O-CNN was submitted to SIGGRAPH in the end of 2016 and was
officially accepted in March, 2017. The camera-ready version of our O-CNN was
submitted to SIGGRAPH in April, 2017. We just did not post our paper on Arxiv
during the review process of SIGGRAPH. Therefore, **the idea of constraining CNN
computation into sparse non-emtpry voxels is first proposed by our O-CNN**.
Currently, this type of 3D convolution is known as Sparse Convolution in the
research community. 

## Key benefits of ocnn-pytorch

- **Simplicity**. The ocnn-pytorch is based on pure PyTorch, it is portable and
  can be intalled with a simple command:`pip install ocnn`. Other sparse
  convolution frameworks heavily rely on C++ and CUDA, and it is complicated to
  configure the compiling environment.

- **Efficiency**. The ocnn-pytorch is very efficient compared with other sparse
  convolution frameworks.  It only takes 18 hours to train the network on
  ScanNet for 600 epochs with 4 V100 GPUs. For reference, under the same
  training settings, MinkowskiNet 0.4.3 takes 60 hours and MinkowskiNet 0.5.4
  takes 30 hours.

