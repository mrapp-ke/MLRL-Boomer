.. _installation:

Installation
============

All algorithms that are provided as part of this project are published on `PyPi <https://pypi.org/>`__. As shown below, they can easily be installed via the Python package manager `pip <https://en.wikipedia.org/wiki/Pip_(package_manager)>`_. Unless you intend to modify the algorithms' source code, in which case you should have a look at the section :ref:`compilation`, this is the recommended way for installing the software.

Installing the BOOMER algorithm
-------------------------------

The BOOMER algorithm is published as the Python package `mlrl-boomer <https://pypi.org/project/mlrl-boomer/>`__. It can be installed via the following command:

.. code-block:: text

   pip install mlrl-boomer

.. note::
    Currently, the above package is available for Linux (x86_64 and aarch64), MacOS (x86_64) and Windows (AMD64).

An example of how to use the algorithm in your own Python program can be found in the section :ref:`usage`.

Installing the Command Line API
-------------------------------

To ease the use of the algorithms that are developed by this project, we also provide a command line API that allows to configure the algorithms and apply them to a given dataset without the need to write code. It is published as the Python package `mlrl-testbed <https://pypi.org/project/mlrl-testbed/>`__ and can optionally be installed via the following command:

.. code-block:: text

   pip install mlrl-testbed

For more information about how to use the command line API, refer to the section :ref:`testbed`.
