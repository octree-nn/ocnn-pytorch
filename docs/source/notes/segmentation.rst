Segmentation
===========================


ScanNet
---------------------------

#. Download the data from the
   `ScanNet benchmark <http://kaldir.vc.in.tum.de/scannet_benchmark/>`_.
   Unzip the data and place it to the folder ``<scannet_folder>``.
   Enter the subdirectory ``projects``, run the following command
   to prepare the dataset.

   .. code-block:: none

      python tools/scannet.py --run process_scannet --path_in <scannet_folder>


#. Run the following command to train the network with 4 GPUs. The mIoU on the
   validation set is 73.6, the training log and weights can be downloaded from
   this `link <https://1drv.ms/u/s!Ago-xIr0OR2-cH_ZcJj2G8G9Naw?e=RhGMOt>`_.

   .. code-block:: none

      python scripts/run_scannet.py --gpu 0,1,2,3 --alias scannet  


#. Run the following command to get the per-point predictions for the validation
   dataset with a voting strategy. And after voting the mIoU is **74.8** on the 
   validation dataset.

   .. code-block:: none

      python scripts/run_scannet.py --run validate --alias scannet


#. To achieve the 76.2 mIoU on the testing set of the 
   `ScanNet benchmark <http://kaldir.vc.in.tum.de/scannet_benchmark>`_,
   run the following command to train the network on both the training and
   validation dataset and get the predictions for the testing dataset.

   .. code-block:: none

      python scripts/run_scannet.py --run train_all --gpu 0,1,2,3 --alias all
      python scripts/run_scannet.py --run test    --alias all

.. note::
    
    The ocnn-pytorch is very efficient compared with other sparse convolution
    frameworks.  It only takes 18 hours to train the network on ScanNet for 600
    epochs with 4 V100 GPUs. For reference, under the same training settings,
    MinkowskiNet 0.4.3 takes 60 hours and MinkowskiNet 0.5.4 takes 30 hours.
