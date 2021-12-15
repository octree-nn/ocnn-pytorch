Usage
=====

.. _installation:

Installation
------------

.. To use Lumache, first install it using pip:

.. .. code-block:: console

..    (.venv) $ pip install lumache

APIs
----------------

Use the ``ocnn.octree.key2xyz`` function:

.. autofunction:: ocnn.octree.key2xyz

.. The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
.. or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
.. will raise an exception.

.. .. autoexception:: lumache.InvalidKindError

.. For example:

.. >>> import lumache
.. >>> lumache.get_random_ingredients()
.. ['shells', 'gorgonzola', 'parsley']

