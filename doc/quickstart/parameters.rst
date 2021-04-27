Parameters
----------

The behavior of the BOOMER algorithm can be controlled in a fine-grained manner via a large number of parameters. Most of these parameters are optional. If not specified otherwise, default settings that work well in most of the cases are used.

In the following an overview of all available parameters is provided.

**Data set**

The following parameters are always needed to specify the data set that should be used for training:

* ``--data-dir``

  * The path of the directory where the data set files are located (an ARFF file and a corresponding XML file according to the Mulan format).

* ``--dataset``

  * The name of the data set files (without suffix).

**Training/Testing Procedure**

* ``--folds`` (default value ``1``)

  * The total number of folds to be used for cross validation or ``1``, if no cross validation should be used.
* ``--current-fold`` (default value ``-1``)

  * The cross-validation fold to be performed or ``-1``, if all folds should be performed. Must be ``-1`` or greater than ``0`` and less or equal to ``--folds``. If ``--folds`` is ``1``, this parameter is ignored.

* ``--evaluate-training-data`` (default value ``False``)

  * ``True``, if the models should not only be evaluated on the test data, but also on the training data, ``False`` otherwise.

**Data Format**

The following parameters allow to specify how the training data should be organized:

* ``--one-hot-encoding`` (default value ``False``)

  * ``True``, if one-hot-encoding should be used for encoding nominal attributes, ``False`` otherwise. One-hot-encoding is not necessary to be able to handle nominal attributes, as the algorithm can handle this kind of attributes natively. 

* ``--feature-format`` (default value ``auto``)

  * ``auto`` The most suitable format for representation of the feature matrix is chosen automatically, based on an estimate of which representation requires less memory.
  * ``dense`` Enforces that the feature matrix is stored using a dense format. 
  * ``sparse`` Enforces that the feature matrix is stored using a sparse format. Using a sparse format may reduce the memory footprint and/or speed up the training process on some data sets.

* ``--label-format`` (default value ``auto``)

  * ``auto`` The most suitable format for representation of the label matrix is chosen automatically, based on an estimate of which representation requires less memory.
  * ``dense`` Enforces that the label matrix is stored using a dense format.
  * ``sparse`` Enforces that the label matrix is stored using a sparse format. Using a sparse format may reduce the memory footprint on some data sets.

**Input Files**

The following parameters allow to specify the directories, where input files can be found:

* ``--model-dir`` (default value ``None``)

  * The path of the directory where saved models are located. If such models are found in the specified directory, they will be used instead of training from scratch. If no models are available, the trained models will be saved in the specified directory once training has completed.

* ``--parameter-dir`` (default value ``None``)

  * The path of the directory where configuration files that provide parameter settings are located. If such files are found in the specified directory, the specified parameter settings are used instead of the parameters that are provided via command line parameters.

**Output**

The following parameters allow to customize the console output and output files that are written by the algorithm:

* ``--output-dir`` (default value ``None``)

  * The path of the directory into which the experimental results should be written.

* ``--store-predictions`` (default value ``False``)

  * ``True``, if the predictions for individual examples and labels should be written into output files, ``False`` otherwise. Does only have an effect if the parameter ``--output-dir`` is specified.

* ``--print-rules`` (default value ``False``)

  * ``True``, if the induced rules should be printed to the console, ``False`` otherwise.

* ``--store-rules`` (default value ``False``)

  * ``True``, if the induced rules should be written to a text file, ``False`` otherwise. Does only have an effect if the parameter ``--output-dir`` is specified.

* ``--print-options`` (default value ``None``)

  * Additional options to be used when writing rules to the console or an output file, if the parameter ``--print-rules`` and/or ``--store-rules`` is set to ``True``. Must be given in the Python dictionary format, e.g. ``{'print_nominal_values':True}``.

* ``--log-level`` (default value ``info``)

  * The log level to be used. Must be ``debug``, ``info``, ``warn``, ``warning``, ``error``, ``critical``, ``fatal`` or ``notset``.


**Algorithmic Parameters**

The following parameters allow to adjust the behavior of the algorithm:

* ``--random-state`` (default value ``1``)

  * The seed to be used by random number generators. Must be at least ``1``.

* ``--max-rules`` (default value ``1000``)

  * The number of rules to be induced or ``-1``, if the number of rules should not be restricted.

* ``--default-rule`` (default value ``True``)

  * ``True``, if the first rule should be a default rule, ``False`` otherwise.

* ``--time-limit`` (default value ``-1``)

  * The duration in seconds after which the induction of rules should be canceled or ``-1``, if no time limit should be set.

* ``--label-sub-sampling`` (default value ``None``)

  * ``None`` All labels are considered for learning a new rule.
  * ``random-label-selection`` The labels to be considered when learning a new rule are chosen randomly. Additional arguments may be provided using the Python dictionary syntax, e.g., ``random-label-selection{'num_samples':5}``.

* ``--feature-sub-sampling`` (default value ``random-feature-selection``)

  * ``None`` All features are considered for learning a new rule.
  * ``random-feature-selection`` A random subset of the features is used to search for the refinements of rules. Additional arguments may be provided using the Python dictionary syntax, e.g., ``random_feature-selection{'sample_size':0.5}``.

* ``--instance-sub-sampling`` (default value ``bagging``)

  * ``None`` All training examples are considered for learning a new rule.
  * ``random-instance-selection`` The training examples to be considered for learning a new rule are selected randomly without replacement. Additional arguments may be provided using the Python dictionary syntax, e.g., ``random-instance-selection{'sample_size':0.5}``.
  * ``bagging`` The training examples to be considered for learning a new rule are selected randomly with replacement. Additional arguments may be provided using the Python dictionary syntax, e.g., ``bagging{'sample_size':0.5}``.
  * ``stratified-label-wise`` The training examples to be considered for learning a new rule are selected according to an iterative stratified sampling method that ensures that for each label the proportion of relevant and irrelevant examples is maintained. Additional arguments may be provided using the Python dictionary syntax, e.g., ``stratified-label-wise{'sample_size':0.5}``.
  * ``stratified-example-wise`` The training examples to be considered for learning a new rule are selected according to stratified sample, where distinct label vectors are treated as individual classes. Additional arguments may be provided using the Python dictionary syntax, e.g., ``stratified-example-wise{'sample_size':0.5}``.

* ``--recalculate-predictions`` (default value ``True``)

  * ``True``, if the predictions of rules should be recalculated on the entire training data, if the parameter ``instance-sub-sampling`` is not set to ``None``, ``False`` otherwise.

* ``--holdout`` (default value ``0``)

  * The fraction of the training examples that should be included in a holdout set. Must be in greater than ``0`` and smaller than ``1`` or ``0``, if no holdout set should be used.

* ``--early-stopping`` (default value ``None``)

  * ``None`` No strategy for early-stopping is used.
  * ``measure`` Stops the induction of new rules as soon as the performance of the model does not improve on a holdout set, according to the loss function. Additional arguments may be provided using the Python dictionary syntax, e.g., ``measure{'min_rules':100,'update_interval':1,'stop_interval':1,'num_past':50,'num_recent':50,'aggregation':'min','tolerance':0.001}``. Does only have an effect if the parameter ``--holdout`` is set to a value greater than ``0``.

* ``--feature-binnig`` (default value ``None``)

  * ``None`` No feature binning is used.
  * ``equal-width`` Examples are assigned to bins, based on their feature values, according to the equal-width binning method. Additional arguments may be provided using the Python dictionary syntax, e.g., ``equal-width{'bin_ratio':0.5,'min_bins':2,'max_bins':256}``.
  * ``equal-frequency``. Examles are assigned to bins, based on their feature values, according to the equal-frequency binning method. Additional arguments may be provided using the Python dictionary syntax, e.g., ``equal-frequency{'bin_ratio':0.5,'min_bins':2,'max_bins':256}``.

* ``--label-binning`` (default value ``None``)

  * ``None`` No label binning is used.
  * ``equal-width`` The labels for which a rule may predict are assigned to bins according to the equal-width binning method. Additional arguments may be provided using the Python dictionary syntax, e.g., ``equal-width{'bin_ratio':0.04,'min_bins':1,'max_bins':8``.

* ``--pruning`` (default value ``None``)

  * ``None`` No pruning is used.
  * ``irep``. Subsequent conditions of rules may be pruned on a holdout set, similar to the IREP algorithm. Does only have an effect if the parameter ``--instance-sub-sampling`` is not set to ``None``.

* ``--min-coverage`` (default value ``1``)

  * The minimum number of training examples that must be covered by a rule. Must be at least ``1``.

* ``--max-conditions`` (default value ``-1``)

  * The maximum number of conditions to be included in a rule's body. Must be at least ``1`` or ``-1``, if the number of conditions should not be restricted.

* ``--max-head-refinements`` (default value ``-1``)

  * The maximum number of times the head of a rule may be refined. Must be at least ``1`` or ``-1``, if the number of refinements should not be restricted.

* ``--head-refinement`` (default value ``None``)

  * ``None`` The most suitable strategy for finding the heads of rules is chosen automatically based on the loss function.
  * ``single-label`` If all rules should predict for a single label.
  * ``full`` If all rules should predict for all labels simultaneously, potentially capturing dependencies between the labels.

* ``--shrinkage`` (default value ``0.3``)

  * The shrinkage parameter, a.k.a. the learning rate, to be used. Must be greater than ``0`` and less or equal to ``1``.

* ``--loss`` (default value ``label-wise-logistic-loss``)

  * ``label-wise-logistic-loss`` A variant of the logistic loss function that is applied to each label individually.
  * ``label-wise-squared-error-loss`` A variant of the Squared error loss that is applied to each label individually.
  * ``label-wise-hinge-loss`` A variant of the Hinge loss that is applied to each label individually.
  * ``example-wise-logistic-loss`` A variant of the logistic loss function that takes all labels into account at the same time.

* ``--predictor`` (default value ``None``)

  * ``None`` The most suitable strategy for making predictions is chosen automatically, depending on the loss function.
  * ``label-wise`` The prediction for an example is determined for each label independently.
  * ``example-wise`` The label vector that is predicted for an example is chosen from the set of label vectors encountered in the training data.

* ``--l2-regularization-weight`` (default value ``1.0``)

  * The weight of the L2 regularization. Must be at least ``0``. If ``0`` is used, the L2 regularization is turned off entirely. Increasing the value causes the model to become more conservative.

**Multithreading**

The following parameters allow to enable multi-threading for different aspects of the algorithm:

* ``--num-threads-refinements`` (default value ``1``)

  * The number of threads to be used to search for potential refinements of rules in parallel. Must be at least ``1`` or ``-1``, if the number of cores that are available on the machine should be used.

* ``--num-threads-update`` (default value ``1``)

  * The number of threads to be used for calculating the gradients and Hessians for different examples in parellel. Must be at least ``1`` or ``-1``, if the number of cores that are available on the machine should be used.

* ``--num-threads-prediction`` (default value ``1``)

  * The number of threads to be used for making predictions for different examples in parallel. Must be at least ``1`` or ``-1``, if the number of cores that are available on the machine should be used.
