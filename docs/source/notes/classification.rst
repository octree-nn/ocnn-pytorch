Classification
===========================


ModelNet40
---------------------------

#. Clone the ``ocnn-pytorch`` repository, and enter the subdirectory
   ``projects``. Run the following command to prepare the dataset.

    .. code-block:: none
    
      python tools/cls_modelnet.py


#. Train the LeNet used in our paper `O-CNN <https://wang-ps.github.io/O-CNN>`_.
   The classification accuracy on the testing set  without voting is **91.7%**.
   And the training log and weights can be downloaded `here
   <https://1drv.ms/u/s!Ago-xIr0OR2-b2WkgDqYEh6EDRw?e=gb1RX4>`_.

    .. code-block:: none

      python classification.py --config configs/cls_m40.yaml SOLVER.alias time


#. Train the HRNet used in our paper on `3D Unsupervised Pretraining
   <https://wang-ps.github.io/pretrain>`_. The classification accuracy on the
   testing set without voting is **93.0%**. And the training log and weights can
   be downloaded `here <https://1drv.ms/u/s!Ago-xIr0OR2-aiT3IUrezwcW7aY?e=ek9ffr>`_.

    .. code-block:: none

        python classification.py --config configs/cls_m40_hrnet.yaml SOLVER.alias time
