Examples
========

The examples use synthetic CARWatch data from ``examples/data`` and do not
depend on the ignored ``playground`` directory.

Run the Python script from the repository root:

.. code-block:: console

   uv run python examples/CARWatch_Import_Example.py

Register the project environment as an IPython kernel before opening the
notebook:

.. code-block:: console

   uv run poe conf_jupyter

Then open ``CARWatch_Import_Example.ipynb`` in a Jupyter frontend and select the
``carwatch`` kernel. The notebook already references this kernelspec.
