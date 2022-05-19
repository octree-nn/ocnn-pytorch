Classification
===========================


Data Preparation
---------------------------

Clone the `ocnn-pytorch` repository, and enter the subdirectory `projects`.

.. code-block:: none

  python tools/cls_modelnet.py


Experiments
---------------------------


#. Train a LeNet used in our paper `O-CNN <https://wang-ps.github.io/O-CNN>`_.

  .. code-block:: none

    python classification.py --config configs/cls_m40.yaml  SOLVER.alias time
  


#. Train a HRNet used in our paper on `3D Unsupervised Pretraining <https://wang-ps.github.io/pretrain>`_

  .. code-block:: none

    python classification.py --config configs/cls_m40_hrnet.yaml  SOLVER.alias time
