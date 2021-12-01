.. _experiments:

Running Experiments
-------------------

In the following, a minimal working example of how to use the command line API for applying the BOOMER algorithm to a particular dataset is shown:

.. code-block:: text

   boomer --data-dir /path/to/datasets/ --dataset name

Both arguments that are included in the above command are mandatory:

* ``--data-dir``: The path of the directory where the data set files are located.
* ``--dataset``: The name of the data set files (without suffix).

The program requires the data set files to be provided in the `Mulan format <http://mulan.sourceforge.net/format.html>`_. It requires two files to be present in the specified directory:

#. An `.arff <http://weka.wikispaces.com/ARFF>`_ file that specifies the feature values and ground truth labels of the training examples.
#. A .xml file that specifies the names of the labels.

This format is commonly used by benchmark datasets that are used for experimental studies in scientific research. A list of publicly available benchmark datasets is provided in the section :ref:`datasets`.
