.. _testbed:

Command Line API
----------------

TODO

.. note::
    Each parameter is identified by an unique name and must be specified according to the following syntax:

    ``--parameter-name value``

    In addition to the specified value, some parameters allow to specify additional options as key-value pairs. These options may be provided by using the following bracket notation:

    ``--parameter-name value{key1=value1,key2=value2}``

    Parameter values that include additional options may not contain any spaces. Depending on the shell that is used to run the program, special characters like ``{`` or ``}`` must eventually be escaped. When using bash or sh this can be achieved by adding single quotes as follows:

    ``--parameter-name value'{key1=value1,key2=value2}'``.

**Data Format**

The following parameters allow to specify how the training data should be organized:

* ``--one-hot-encoding`` (Default value = false)

  * ``true`` One-hot-encoding is used to encode nominal attributes.
  * ``false`` The algorithm's ability to natively handle nominal attributes is used.

**Data set**

The following parameters are always needed to specify the data set that should be used for training:

* ``--data-dir``

  * The path of the directory where the data set files are located (an ARFF file and a corresponding XML file according to the Mulan format).

* ``--dataset``

  * The name of the data set files (without suffix).

**Training/Testing Procedure**

* ``--folds`` (Default value = 1)

  * The total number of folds to be used for cross validation. Must be greater than 1 or 1, if no cross validation should be used.

* ``--current-fold`` (Default value = 0)

  * The cross validation fold to be performed. Must be in [1, --folds] or 0, if all folds should be performed. This parameter is ignored if --folds is set to 1.

* ``--evaluate-training-data`` (Default value = false)

  * ``true`` The models are not only evaluated on the test data, but also on the training data.
  * ``false`` The models are only evaluated on the test data.

**Input Files**

The following parameters allow to specify the directories, where input files can be found:

* ``--model-dir`` (Default value = None)

  * The path of the directory where models should be stored. If such models are found in the specified directory, they will be used instead of training from scratch. If no models are available, the trained models will be saved in the specified directory once training has completed.

* ``--parameter-dir`` (Default value = None)

  * The path of the directory where configuration files, which specify the parameters to be used by the algorithm, are located. If such files are found in the specified directory, the specified parameter settings are used instead of the parameters that are provided via command line parameters.

**Output**

The following parameters allow to customize the console output and output files that are written by the algorithm:

* ``--output-dir`` (Default value = None)

  * The path of the directory where experimental results should be saved.

* ``--print-evaluation`` (Default value = true)

  * ``true`` The evaluation results in terms of common metrics are printed on the console.
  * ``false`` The evaluation results are not printed on the console.

* ``--store-evaluation`` (Default value = true)

  * ``true`` The evaluation results in terms of common metrics are written into output files. Does only have an effect if the parameter --output-dir is specified.
  * ``false`` The evaluation results are not written into output files.

* ``--store-predictions`` (Default value = false)

  * ``true`` The predictions for individual examples and labels are written into output files. Does only have an effect if the parameter --output-dir is specified.
  * ``false`` Predictions are not written into output files.

* ``--print-data-characteristics`` (Default value = false)

  * ``true`` The characteristics of the training data set are printed on the console
  * ``false`` The characteristics of the training data set are not printed on the console

* ``--store-data-characteristics`` (Default value = false)

  * ``true`` The characteristics of the training data set are written into a CSV file. Does only have an effect if the parameter --output-dir is specified.
  * ``false`` The characteristics of the training data set are not written into a CSV file.

* ``--print-model-characteristics`` (Default value = false)

  * ``true`` The characteristics of rule models are printed on the console
  * ``false`` The characteristics of rule models are not printed on the console

* ``--store-model-characteristics`` (Default value = false)

  * ``true`` The characteristics of rule models are written into a CSV file. Does only have an effect if the parameter --output-dir is specified.
  * ``false`` The characteristics of rule models are not written into a CSV file.

* ``--print-rules`` (Default value = false)

  * ``true`` The induced rules are printed on the console.
  * ``false`` The induced rules are not printed on the console.

* ``--store-rules`` (Default value = false)

  * ``true`` The induced rules are written into a text file. Does only have an effect if the parameter --output-dir is specified.
  * ``false`` The induced rules are not written into a text file.

* ``--print-options`` (Default value = None)

  * Additional options to be taken into account when writing rules on the console or into an output file. Does only have an effect, if the parameter --print-rules or --store-rules is set to ``true``. The options must be given using the bracket notation. The following options are available:

    * ``print_feature_names`` (Default value = true) ``true``, if the names of features should be printed instead of their indices, ``false`` otherwise.
    * ``print_label_names`` (Default value = true) ``true``, if the names of labels should be printed instead of their indices, ``false`` otherwise.
    * ``print_nominal_values`` (Default value = true) ``true``, if the names of nominal values should be printed instead of their numerical representation, ``false`` otherwise.

* ``--log-level`` (Default value = info)

  * The log level to be used. Must be debug, info, warn, warning, error, critical, fatal or notset.
