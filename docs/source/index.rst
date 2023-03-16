:github_url: https://github.com/octree-nn/ocnn-pytorch

ocnn-pytorch
=================

ocnn-pytorch is a **pure-PyTorch**-based implementation of `O-CNN <https://wang-ps.github.io/O-CNN>`_.
The code has been tested with `Pytorch>=1.6.0`, and `Pytorch>=1.9.0` is preferred.

O-CNN is an octree-based sparse convolutional neural network framework for 3D deep learning. O-CNN constrains the CNN storage and computation into non-empty sparse voxels for efficiency and uses the `octree` data structure to organize and index these sparse voxels.
The concept of sparse convolution in O-CNN is the same with `H-CNN <https://ieeexplore.ieee.org/abstract/document/8580422>`_, `SparseConvNet <https://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf>`_, and `MinkowskiNet <https://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf>`_.
The key difference is that our O-CNN uses the `octree` to index the sparse voxels, while these 3 works use the `Hash Table`.

Our O-CNN is published in SIGGRAPH 2017, H-CNN is published in TVCG 2018, SparseConvNet is published in CVPR 2018, and MinkowskiNet is published in CVPR 2019.
Actually, our O-CNN was submitted to SIGGRAPH in the end of 2016 and was officially accepted in March, 2017. The camera-ready version of our O-CNN was submitted to SIGGRAPH in April, 2017. We just did not post our paper on Arxiv during the review process of SIGGRAPH.
Therefore, **the idea of constraining CNN computation into sparse non-emtpry voxels is first proposed by our O-CNN**.
Currently, this type of 3D convolution is known as Sparse Convolution in the research community.


.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Notes

  notes/installation
  notes/classification
  notes/segmentation
  notes/autoencoder
  notes/completion


.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Package Reference

  modules/octree
  modules/nn
  modules/modules
  modules/models
  modules/dataset
  modules/utils


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
