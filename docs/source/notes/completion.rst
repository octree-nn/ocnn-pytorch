Completion
===========================


ShapeNet
---------------------------

#. Download the dataset for training and testing. The dataset is used by our
   paper on `shape completion <https://arxiv.org/abs/2006.03762>`__, which
   contains point clouds sampled from  meshes of 8 categories in 
   `ShapeNet <https://shapenet.org/>`__. The point clouds are in the format of
   `ply`, which can be visualized via viewers like meshlab. Clone the
   ``ocnn-pytorch`` repository, and enter the subdirectory ``projects``, then
   run the following command.

   .. code-block:: none

      python tools/ae_shapenet.py --run prepare_dataset


#. Run the following command to train the network. The training log and weights
   can be downloaded `here <todo>`__.

   .. code-block:: none

      python completion.py --config configs/completion.yaml


.. #. Run the following command to get the predictions on the testing dataest. The 
..    parameter following ``SOLVER.ckpt`` can be freely modified to test different
..    trained weights. And the results are in the folder ``logs/ae_shapenet/ae_eval``.

..    .. code-block:: none

..       python autoencoder.py --config configs/ae_shapenet.yaml             \
..              SOLVER.run evaluate  SOLVER.alias eval                       \
..              SOLVER.ckpt logs/ae_shapenet/ae/checkpoints/00300.model.pth

