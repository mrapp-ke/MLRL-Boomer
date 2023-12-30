.. _evaluation:

Performance Evaluation
======================

A major task in machine learning is to assess the predictive performance of different learning approaches, compare them to each other, and decide for the best approach suitable for a particular problem. The command line API provided by this project helps with these tasks by implementing several strategies for splitting available data into training and test sets, which is crucial to obtain unbiased estimates of a method's performance. In accordance with established practices, a machine learning model that is trained on a test set is afterwards applied to the corresponding test set to obtain predictions for data that was not included in the training process. The metrics that are used for evaluating the quality of these predictions are automatically chosen, depending on the type of predictions (binary predictions, probability estimates, etc.) provided by the tested method.

Strategies for Data Splitting
-----------------------------

Several strategies for splitting the available data into distinct training and test sets can be used via the command line API. They are described in the following.

Train-Test-Splits
^^^^^^^^^^^^^^^^^

The simplest and computationally least demanding strategy for obtaining training and tests is to randomly split the available data into two, mutually exclusive, parts. This strategy, which is used by default, if not specified otherwise, can be used by providing the argument ``--data-split train-test`` to the command line API:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --data-split train-test

Following the argument ``--dataset``, the program will load the training data from a file named ``dataset-name_training.arff``. Similarly, it will expect the test data to be stored in a file named ``dataset-name_test.arff``. If these files are not available, the program will look for a file with the name ``dataset-name.arff`` and split it into training and test data automatically.

When it is the responsibility of the command line API to split a given dataset into training and test tests, 66% of the data will be included in the training set, whereas the remaining 33% will be part of the test set. Although this ratio is frequently used in machine learning, you can easily adjust it by providing the option ``test_size``:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --data-split 'train-test{test_size=0.25}'

This command will tell the command line API to include 75% of the available data in the training set and use the remaining 25% for the test set.

Cross Validation
^^^^^^^^^^^^^^^^

A more elaborate strategy for splitting data into training and test sets, which results in more realistic performance estimates, but also entails greater computational costs, is referred to as `cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`__ (CV). The basic idea is to split the available data into several, equally-sized, parts. Afterwards, several machine learning models are trained and evaluated on different portions of the data using the same learning method. Each of these parts will be used for testing exactly once, whereas the remaining ones make up the training set. The performance estimates that are obtained for each of these subsequent runs, referred to as *folds*, are finally averaged to obtain a single score and corresponding `standard deviation <https://en.wikipedia.org/wiki/Standard_deviation>`__. The command line API can be instructed to perform a cross validation using the argument ``--data-split cv``:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --data-split cv

By default, a 10-fold cross validation, where ten models are trained and evaluated, will be performed. The number of folds can easily be adjusted via the option ``num_folds``. For example, the following command results in a 5-fold CV being used:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --data-split 'cv{num_folds=5}'

.. tip::
    When providing the option ``current_fold``, only a single fold, instead of the entire procedure, will be performed. This is particularly useful, if one intends to train and evaluate the models for each individual fold in parallel on different machines. For example, the following command does only execute the second fold of a 5-fold CV:

    .. code-block:: text

       boomer --data-dir /path/to/datsets/ --dataset dataset-name --data-split 'cv{num_folds=5,current_fold=2}'

Evaluation on the Training Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
    The configuraton described in this section should only be used for testing purposes, as the evaluation results will be highly biased and overly optimistic.

Sometimes, evaluating the performance of a model on the data it has been trained on can be helpful for analyzing the behavior of a machine learning algorithm, e.g., if one needs to check if the approach is able to fit the data accurately. For this purpose, the command line API allows to use the argument ``--data-split none``, which will not result in the given data to be split at all. Instead, the learning algorithm will be applied to the entire dataset and predictions will be obtained from the resulting model for the exact same data points. The argument can be specified as follows:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --data-split none

.. tip::
    If you are interested in obtaining evaluation results for the training data in addition to the test data when using a train-test-split or a cross validation, as discussed above, the argument ``--evaluate-training-data true`` may be used:

    .. code-block:: text

       boomer --data-dir /path/to/datsets/ --dataset dataset-name --data-split cv --evaluate-training-data true
