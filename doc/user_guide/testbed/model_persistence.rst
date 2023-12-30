.. _model_persistence:

Saving and Loading Models
=========================

Because the training of machine learning models can be time-consuming, they are usually trained once and then reused later for making predictions. For this purpose, the command line API provides means to store models on disk and load them from the created files later on. This requires to specify the path of a directory, where models should be saved, via the command line argument ``--model-dir``:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --model-dir /path/to/models

.. note::
    The path of the directory, where models should be saved, can be either absolute or relative to the working directory.

If :ref:`train_test_split` are used for evaluating the predictive performance of models, a single model will be fit to the training data and stored in a file:

* ``boomer.model``

If a :ref:`cross_validation` is performed instead, one model is trained per cross validation fold and all of these models are stored in the specified directory. For example, a 5-fold cross validation will result in the following files:

* ``boomer_fold-1.model``
* ``boomer_fold-2.model``
* ``boomer_fold-3.model``
* ``boomer_fold-4.model``
* ``boomer_fold-5.model``

When executing the aforementioned command again, the program will recognize the previously stored models in the specified directory. Instead of training them from scratch, the models will then be loaded from the respective files, which should be much faster than training them again.
