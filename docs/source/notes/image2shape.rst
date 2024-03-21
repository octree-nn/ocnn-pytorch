Image2Shape
===========================


ShapeNet
---------------------------

#. Download the dataset for training and testing `here <https://www.dropbox.com/scl/fi/fxxpdc8pnxjjrnkw1073v/ShapeNetV1.zip?rlkey=1erwximzjmy87ssk30c1gdded&dl=0>`__.
   Clone the  latest ``ocnn-pytorch`` repository, and enter the subdirectory
   ``projects``. Unzip and place the data in the folder ``data/ShapeNetV1``.


#. Run the following command to train the network. The training log and weights
   can be downloaded `here <https://www.dropbox.com/scl/fi/yriskwzgfj99l1wdej2sj/image2shape_trained_models.zip?rlkey=9now3gbkplewd4319fo8jzyvq&dl=0>`__. The training process takes 9 hours on 4
   Nvidia 2080 GPUs.

   .. code-block:: none

      python image2shape.py --config configs/image2shape.yaml SOLVER.gpu 0,1,2,3


#. Run the following command to get the predictions on the testing dataest. The
   parameter following ``SOLVER.ckpt`` can be freely modified to test different
   trained weights. And the results are in the folder ``logs/image2shape/eval``.

   .. code-block:: none

      python image2shape.py --config configs/image2shape.yaml             \
             SOLVER.run evaluate  SOLVER.alias eval                       \
             SOLVER.ckpt logs/image2shape/image2shape/checkpoints/00300.model.pth

