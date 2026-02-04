:github_url: https://github.com/octree-nn/ocnn-pytorch

ocnn-pytorch
=================

.. image:: https://readthedocs.org/projects/ocnn-pytorch/badge/?version=latest
   :target: https://ocnn-pytorch.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://static.pepy.tech/badge/ocnn
   :target: https://pepy.tech/project/ocnn
   :alt: Downloads

.. image:: https://static.pepy.tech/badge/ocnn/month
   :target: https://pepy.tech/project/ocnn
   :alt: Downloads (Monthly)

.. image:: https://img.shields.io/pypi/v/ocnn
   :target: https://pypi.org/project/ocnn/
   :alt: PyPI

This repository contains the **pure PyTorch**-based implementation of `O-CNN
<https://wang-ps.github.io/O-CNN.html>`_. The code is compatible with
`Pytorch>=1.6.0`, while `Pytorch>=2.8.0` is required for Triton-based convolutions.
The *original* implementation of O-CNN is based on C++ and CUDA and can be found
`here <https://github.com/Microsoft/O-CNN>`_, which has received over `730
stars`.


O-CNN is an octree-based 3D convolutional neural network framework for 3D data.
O-CNN constrains the CNN storage and computation into non-empty sparse voxels
for efficiency and uses the `octree` data structure to organize and index these
sparse voxels. Currently, this type of 3D convolution is known as Sparse
Convolution in the research community.

The concept of Sparse Convolution in O-CNN is the same as
`SparseConvNet <https://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf>`_,
`MinkowskiNet <https://github.com/NVIDIA/MinkowskiEngine>`_, and
`SpConv <https://github.com/traveller59/spconv>`_.
The key difference is that our O-CNN uses `octrees` to index the sparse voxels,
while these works use `Hash Tables`. However, I believe that `octrees` may be
the right choice for Sparse Convolution. With `octrees`, I can implement the
Sparse Convolution with pure PyTorch. More importantly, with `octrees`, I can
also build efficient transformers for 3D data --
`OctFormer <https://github.com/octree-nn/octformer>`_, which is extremely hard
with `Hash Tables`.

Our O-CNN is published in SIGGRAPH 2017, SparseConvNet is published in CVPR
2018, and MinkowskiNet is published in CVPR 2019. Actually, our O-CNN was
submitted to SIGGRAPH at the end of 2016 and was officially accepted in March
2017.
We just did not post our paper on Arxiv during the review process of SIGGRAPH.
Therefore, **the idea of constraining CNN computation into sparse non-empty
voxels, i.e. Sparse Convolution, is first proposed by our O-CNN**.


Key benefits of ocnn-pytorch include:

- **Simplicity**. The ocnn-pytorch is based on pure PyTorch, it is portable and
  can be installed with a simple command:pip install ocnn. Other sparse
  convolution frameworks heavily rely on C++ and CUDA, and it is complicated to
  configure the compiling environment.

- **Efficiency**. The ocnn-pytorch is very efficient compared with other sparse
  convolution frameworks. It only takes 18 hours to train the network on ScanNet
  for 600 epochs with 4 V100 GPUs. For reference, under the same training
  settings, MinkowskiNet 0.4.3 takes 60 hours and MinkowskiNet 0.5.4 takes 30
  hours.

.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Notes

  notes/installation
  notes/classification
  notes/segmentation
  notes/autoencoder
  notes/image2shape
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
