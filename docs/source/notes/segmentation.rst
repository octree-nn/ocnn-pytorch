Segmentation
===========================


ScanNet
---------------------------

#. Download the data from the
   `ScanNet benchmark <http://kaldir.vc.in.tum.de/scannet_benchmark/>`__.
   Unzip the data and place it to the folder ``<scannet_folder>``.
   Enter the subdirectory ``projects``, run the following command
   to prepare the dataset.

   .. code-block:: none

      python tools/seg_scannet.py --run process_scannet --path_in <scannet_folder>


#. Run the following command to train the network with 4 GPUs. The mIoU on the
   validation set is 73.6, the training log and weights can be downloaded from
   this `link <https://1drv.ms/u/s!Ago-xIr0OR2-cH_ZcJj2G8G9Naw?e=RhGMOt>`__.

   .. code-block:: none

      python scripts/run_seg_scannet.py --gpu 0,1,2,3 --alias scannet


#. Run the following command to get the per-point predictions for the validation
   dataset with a voting strategy. And after voting the mIoU is **74.8** on the
   validation dataset.

   .. code-block:: none

      python scripts/run_seg_scannet.py --run validate --alias scannet


#. To achieve the 76.2 mIoU on the testing set of the
   `ScanNet benchmark <http://kaldir.vc.in.tum.de/scannet_benchmark>`__,
   run the following command to train the network on both the training and
   validation dataset and get the predictions for the testing dataset.

   .. code-block:: none

      python scripts/run_seg_scannet.py --run train_all --gpu 0,1,2,3 --alias all
      python scripts/run_seg_scannet.py --run test --alias all

.. note::

    The ocnn-pytorch is very efficient compared with other sparse convolution
    frameworks.  It only takes 18 hours to train the network on ScanNet for 600
    epochs with 4 V100 GPUs. For reference, under the same training settings,
    MinkowskiNet 0.4.3 takes 60 hours and MinkowskiNet 0.5.4 takes 30 hours.



SemanticKITTI
---------------------------

#. Download the dataset and semantic labels from the official website of
   `SemanticKITTI <http://www.semantic-kitti.org/dataset.html#download>`__,
   including `data_odometry_velodyne.zip <http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip>`__
   and `data_odometry_labels.zip <http://www.semantic-kitti.org/assets/data_odometry_labels.zip>`__,
   and place them into the folder ``projects/data/SemanticKITTI``. Then enter
   the subdirectory ``projects``, run the following command to prepare the
   dataset.

   .. code-block:: none

      python tools/seg_kitti.py


#. Run the following command to train the network with 4 GPUs. The mIoU on the
   validation set is **64.0**, the training log and weights can be downloaded from
   this `link <https://1drv.ms/u/s!Ago-xIr0OR2-eyisuXI6_Fh0Rrg?e=woPcl9>`__.
   .. I observe random fluctuations of 2 points mIoU in this experiment.

   .. code-block:: none

      python segmentation.py --config configs/seg_kitti.yaml SOLVER.gpu 0,1,2,3


ShapeNet
---------------------------


#. Run the following command to prepare the dataset.

   .. code-block:: none

      python tools/seg_shapenet.py


#. Run the following command to train the a shallow SegNet with an octree depth
   of 5 used in the original experiment of
   `O-CNN <https://wang-ps.github.io/O-CNN.html>`__.
   And the segmentation refinement based on CRF in the paper is omitted for
   simplicity. The category mIoU and instance mIoU without voting is 82.5 and
   84.0 respectively, the training log and weights can be downloaded from this
   `link <https://1drv.ms/u/s!Ago-xIr0OR2-cXkHyzrqrgT-CTo?e=GE0pXi>`__.
   It is also easy to do experiments with an octree depth of 6 by specifying
   command line parameter ``--depth 6``, with which the category mIoU and
   instance mIoU is 82.8 and 84.2 respectively.

   .. code-block:: none

      python scripts/run_seg_shapenet.py --depth 5 --model segnet --alias segnet_d5


#. Run the following command to train the a deep UNet with an octree depth of 5.
   The category mIoU and instance mIoU without voting is **84.2** and **85.4**
   respectively, the training log and weights can be downloaded from this
   `link <https://1drv.ms/u/s!Ago-xIr0OR2-cgSYpuccOEaCmUU?e=guhj1T>`__.

   .. code-block:: none

      python scripts/run_seg_shapenet.py --depth 5 --model unet --alias unet_d5
