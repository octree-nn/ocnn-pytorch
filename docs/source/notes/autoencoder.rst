AutoEncoder
===========================


ShapeNet
---------------------------

#. Download the dataset for training and testing `here <https://www.dropbox.com/scl/fi/fxxpdc8pnxjjrnkw1073v/ShapeNetV1.zip?rlkey=1erwximzjmy87ssk30c1gdded&dl=0>`__.
   Clone the  latest ``ocnn-pytorch`` repository, and enter the subdirectory
   ``projects``. Unzip and place the data in the folder ``data/ShapeNetV1``.

   .. The dataset is used by our
   .. paper on `shape completion <https://arxiv.org/abs/2006.03762>`__, which
   .. contains point clouds sampled from  meshes of 8 categories in
   .. `ShapeNet <https://shapenet.org/>`__. The point clouds are in the format of
   .. `ply`, which can be visualized via viewers like meshlab. Clone the
   .. ``ocnn-pytorch`` repository, and enter the subdirectory ``projects``, then
   .. run the following command.

   .. .. code-block:: none

   ..    python tools/ae_shapenet.py --run prepare_dataset


#. Run the following command to train the network. The training log and weights
   can be downloaded `here <https://1drv.ms/u/s!Ago-xIr0OR2-eSg3Qxu1oNUo9ZY?e=vibpol>`__.

   .. code-block:: none

      python autoencoder.py --config configs/ae_shapenet.yaml


#. Run the following command to get the predictions on the testing dataest. The
   parameter following ``SOLVER.ckpt`` can be freely modified to test different
   trained weights. And the results are in the folder ``logs/ae_shapenet/ae_eval``.

   .. code-block:: none

      python autoencoder.py --config configs/ae_shapenet.yaml             \
             SOLVER.run evaluate  SOLVER.alias eval                       \
             SOLVER.ckpt logs/ae_shapenet/ae/checkpoints/00300.model.pth

