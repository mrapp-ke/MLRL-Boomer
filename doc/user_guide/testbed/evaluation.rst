.. _evaluation:

Performance Evaluation
======================

A major task in machine learning is to assess the predictive performance of different learning approaches, compare them to each other, and decide for the best approach suitable for a particular problem. The command line API provided by this project helps with these tasks by implementing several strategies for splitting available data into training and test sets, which is crucial to obtain unbiased estimates of a method's performance. In accordance with established practices, a machine learning model that is trained on a test set is afterwards applied to the corresponding test set to obtain predictions for data that was not included in the training process. The metrics that are used for evaluating the quality of these predictions are automatically chosen, depending on the type of predictions (binary predictions, probability estimates, etc.) provided by the tested method.

Strategies for Data Splitting
-----------------------------

Several strategies for splitting the available data into distinct training and test sets can be used via the command line API. They are described in the following.

.. _train_test_split:

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

.. _cross_validation:

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

Types of Predictions
--------------------

The metrics for evaluating the quality of predictions that have been obtained for a test set are chosen automatically, depending on the type of predictions provided by the model. In general, the command line API supports evaluating binary predictions, probability estimates, and regression scores. Not all of these prediction types must be supported by a single machine learning method. For example, in case of the BOOMER algorithm, the prediction of probabilities is only possible with certain configurations.

.. _regression_scores:

Regression Scores
^^^^^^^^^^^^^^^^^

We refer to real-valued predictions, which may be positive or negative, as *regression scores*. In the context of multi-label classification, positive scores indicate a preference towards predicting a label as relevant, whereas negative scores are predicted for labels that are more likely to be irrelevant. The absolute size of the scores corresponds to the confidence of the predictions, i.e., if a large value is predicted for a label, the model is more certain about the correctness of the predicted outcome. Unlike :ref:`probability_estimates`, regression scores are not bound to a certain interval and can be arbitrary positive or negative values. The BOOMER algorithm uses regression scores as a basis for predicting probabilities or binary labels. If you want to evaluate the quality of the regression scores directly, instead of transforming them into probabilities or binary predictions, the argument ``--prediction-type scores`` may be passed to the command line API:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --prediction-type scores

For evaluating the quality of regression scores, `multi-label ranking measures <https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics>`__ provided by the `scikit-learn <https://scikit-learn.org>`__ framework are used.

.. _probability_estimates:

Probability Estimates
^^^^^^^^^^^^^^^^^^^^^
Probability estimates are given as real values between zero and one. In the context of multi-label classification, they express the probability of a label being relevant. If the predicted probability is close to zero, the corresponding label is more likely to be irrelevant, whereas a probability close to one is predicted for labels that are likely to be relevant. If you intend to evaluate the quality of probabilistic predictions, the argument ``--prediction-type probabilities`` should be used:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --prediction-type probabilities

Similar to :ref:`regression_scores`, the command line API relies on `multi-label ranking measures <https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics>`__, as implemented by the `scikit-learn <https://scikit-learn.org>`__ framework, for evaluating probability estimates.

Binary Labels
^^^^^^^^^^^^^

The most common type of prediction used for multi-label classification are binary predictions that directly indicate whether a label is considered as irrelevant or relevant. Irrelevant labels are represented by the value ``0``, whereas the value ``1`` is predicted for relevant labels. By default, the command line API instructs the learning method to provide binary predictions. If you want to explicitly instruct it to use this particular type of predictions, you can use the argument ``--prediction-type binary``:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --prediction-type binary

In a multi-label setting, the quality of binary predictions is assessed in terms of commonly used `multi-label classification metrics <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics>`__ implemented by the `scikit-learn <https://scikit-learn.org>`__ framework. If a dataset contains only a single label, the evaluation will be restricted to classification metrics that are suited for single-label classification problems.

Incremental Evaluation
----------------------

When evaluating the predictive performance of an `ensemble method <https://en.wikipedia.org/wiki/Ensemble_learning>`__, i.e., models that consist of several weak predictors, also referred to as *ensemble members*, the command line API supports to evaluate these models incrementally. In particular, rule-based machine learning algorithms like the ones implemented by this project are often considered as ensemble methods, where each rule in a model can be viewed as a weak predictor.  Adding more rules to a model typically results in better predictive performance. However, adding too many rules may result in overfitting the training data and therefore achieving subpar performance on the test data. For analyzing such behavior, the arugment ``--incremental-evaluation true`` may be passed to the command line API:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --incremental-evaluation true

When using the above command, the rule-based model that is learned by the BOOMER algorithm will be evaluated repeatedly as more rules are added to it. Evaluation results will be obtained for a model consisting of a single rule, two rules, three rules, and so on. Of course, because the evaluation is performed multiple times, this evaluation strategy comes with a large computational overhead. Therefore, depending on the size of the final model, it might be necessary to limit the number of evaluations via the following options:

* ``min_size`` specifies the minimum number of ensemble members that must be included in a model for the first evaluation to be performed.
* ``max_size`` specifies the maximum number of ensemble members to be evaluated.
* ``step_size`` allows to to specify after how many additional ensemble members the evaluation should be repeated.

For example, the following command may be used for the incremental evaluation of a BOOMER model that consists of up to 1000 rules. The model will be evaluated for the first time after 200 rules have been added. Subsequent evaluations will be perfomed when the model comprises 400, 600, 800, and 1000 rules.

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --incremental-evaluation 'true{min_size=200,max_size=1000,step_size=200}'
