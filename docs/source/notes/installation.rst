Installation
============


Requirement
---------------------------

Install `PyTorch >= 1.9.0 <https://pytorch.org/get-started/locally/>`_ according
the official PyTorch instructions.


Installation via Pip
---------------------------

Run the following commands to install `ocnn` given PyTorch has been installed.

.. code-block:: none

    pip install ocnn


Installation from Source
---------------------------

It is also easy to install `ocnn` from source.


#. Clone the code:

    .. code-block:: none

      git clone https://github.com/octree-nn/ocnn-pytorch.git

#. Enter the folder containing the code, and run the following command:

    .. code-block:: none

      pip install .


.. note::

    The Triton-based convolution operations in `ocnn` require `PyTorch ≥ 2.9.0`
    and the `triton` package. On Ubuntu systems, the `triton`    package is
    automatically installed during the `PyTorch` installation process. However,
    on Windows systems, the `triton` package is not installed automatically, so
    you need to install it manually using the following command: `pip install
    triton-windows`.