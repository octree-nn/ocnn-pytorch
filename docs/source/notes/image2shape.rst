Image2Shape
===========================


ShapeNet
---------------------------

#. Download the dataset for training and testing.


#. Run the following command to train the network. The training log and weights
   can be downloaded `here <todo>`__.

   .. code-block:: none

      python image2shape.py --config configs/image2shape.yaml


#. Run the following command to get the predictions on the testing dataest. The
   parameter following ``SOLVER.ckpt`` can be freely modified to test different
   trained weights. And the results are in the folder ``logs/image2shape/eval``.

   .. code-block:: none

      python image2shape.py --config configs/image2shape.yaml             \
             SOLVER.run evaluate  SOLVER.alias eval                       \
             SOLVER.ckpt logs/image2shape/ae/checkpoints/00300.model.pth

