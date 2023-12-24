.. _testbed:

Using the Command Line API
==========================

As an alternative to using the BOOMER algorithm in your own Python program (see :ref:`usage`), the command line API that is provided by the package `mlrl-testbed <https://pypi.org/project/mlrl-testbed/>`__ (see :ref:`installation`) can be used to run experiments without the need to write code. Currently, it provides the following functionalities:

* The predictive performance in terms of commonly used evaluation measures can be assessed by using predefined splits of a dataset into training and test data or via `cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_.
* Experimental results can be written into output files. This includes evaluation scores, the predictions of a model, textual representations of rules, as well as the characteristics of models or datasets.
* Models can be stored on disk and reloaded for later use.

Running Experiments
-------------------

In the following, a minimal working example of how to use the command line API for applying the BOOMER algorithm to a particular dataset is shown:

.. code-block:: text

   boomer --data-dir /path/to/datasets/ --dataset name

Both arguments that are included in the above command are mandatory:

* ``--data-dir``: An absolute or relative path to the directory where the data set files are located.
* ``--dataset``: The name of the data set files (without suffix).

The program expects the data set files to be provided in the `Mulan format <http://mulan.sourceforge.net/format.html>`_. It requires two files to be present in the specified directory:

#. An `.arff <http://weka.wikispaces.com/ARFF>`_ file that specifies the feature values and ground truth labels of the training examples.
#. An .xml file that specifies the names of the labels.

The Mulan dataset format is commonly used for benchmark datasets that allow to compare the performance of different machine learning approaches in empirical studies. A collection of publicly available benchmark datasets is available `here <https://github.com/mrapp-ke/Boomer-Datasets>`_.

If an .xml file is not provided, the program tries to retrieve the number of labels from the `@relation` declaration that is contained in the .arff file, as it is intended by the `MEKA project's dataset format <https://waikato.github.io/meka/datasets/>`_. According to the MEKA format, the number of labels may be specified by including the substring "-C L" in the `@relation` name, where "L" is the number of leading attributes in the dataset that should be treated as labels.

Optional Arguments
------------------

In addition to the mandatory arguments that must be provided to the command line API for specifying the dataset used for training, a wide variety of optional arguments for customizing the program's behavior are available as well. An overview of all available command line arguments is provided in the section :ref:`arguments`. For example, they can be used to specify an output directory, where experimental results should be stored:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset name --output-dir /path/to/output/

Moreover, algorithmic parameters that control the behavior of the machine learning algorithm can be set via command line arguments as well. For example, as shown in the section :ref:`setting_algorithmic_parameters`, the value of the parameter ``feature_binning`` can be specified as follows:

.. code-block:: text

   boomer --data-dir /path/to/datasets/ --dataset name --feature-binning equal-width

Some algorithmic parameters, including the parameter ``feature_binning``, come with additional options in the form of key-value pairs. They can be specified by using a bracket notation as shown below:

.. code-block:: text

   boomer --data-dir /path/to/datasets/ --dataset name --feature-binning equal-width'{bin_ratio=0.33,min_bins=2,max_bins=64}'
