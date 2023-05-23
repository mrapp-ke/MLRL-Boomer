# Release Notes

## Version 0.9.0 (to be released)

***This release comes with changes to the command line API. For an updated overview of the available parameters, please refer to the [documentation](https://mlrl-boomer.readthedocs.io).***

A major update to the BOOMER algorithm that introduces the following changes:

* Sparse data structures can now be used to store gradients and Hessians if supported by the loss function. The desired behavior can be specified via a new parameter `--statistic-format`.
* The parameter `--head-type` does now allow to learn partial rules that predict for a predefined number of labels (`partial-fixed`) or a subset of the available labels that is determined dynamically (`partial-dynamic`).
* The parameter `--rule-induction` does now allow to use a top-down beam search for the induction of individual rules (`top-down-beam-search`).
* The parameter `--classification-predictor` has been renamed to `--binary-predictor`. It does now allow to obtain predictions that maximize the example-wise F1-measure (`gfm`).
* The parameter `--loss` does now allow to use variants of the squared error loss and squared hinge loss that take all labels of an example into account at the same time (`squared-error-example-wise` and `squared-hinge-example-wise`).  
* A new parameter `--sequential-post-optimization` has been added. It allows to optimize each rule in a previously learned model by being relearned in the context of the other rules.
* A new parameter `--probability-predictor` has been added. It allows to specify whether probability estimates should be obtained for each label independently (`label-wise`) or via marginalization over the label vectors that are encountered in the training data (`marginalized`).
* The arguments `--print-prediction-characteristics` and `--store-prediction-characteristics` have been added to the command line API. They allow to print certain characteristics of binary predictions or write them into output files.
* The argument ``--incremental-prediction`` has been added to the command line API. It allows to specify whether ensemble models should be evaluated repeatedly, using only a subset of the ensemble members with increasing size, or not.
* Fixed behavior of the argument `--label-format` when set to the value `auto`.
* Rules are now guaranteed to not cover more examples than specified via the parameter `min_coverage`. The parameter is now also taken into account when using feature binning. Alternatively, the minimum coverage of rules can now also be specified as a fraction via the parameter `min_support`. 
* The parameters `--folds` and `--current-fold` have been replaced with a new parameter `--data-split` that provides more control of how data is split into training and test sets.
* The parameter `--predict-probabilities` has been replaced with a new parameter `--prediction-type` that allows to specify whether binary labels (`binary`), regression scores (`scores`) or probability estimates (`probabilities`) should be predicted.
* The parameters `--print-evaluation` and `--store-evaluation` do now allow to specify whether individual evaluation measures should be used or not.
* The parameter `--early-stopping` has been replaced with a new parameter `--global-pruning`. It allows to use early stopping (`pre-pruning`) oder post-pruning (`post-pruning`). 
* The parameter `--pruning` has been renamed to `--rule-pruning`.
* The parameter `--predicted-label-format` has been renamed to `--prediction-format`.
* Data sets in the MEKA format can now be used for experiments.

## Version 0.8.2 (Apr. 11th, 2022)

A bugfix release that solves the following issues:

* Fixed prebuilt packages available at [PyPI](https://pypi.org/project/mlrl-boomer/).
* Fixed output of nominal values when using the option `--print-rules true`.

## Version 0.8.1 (Mar. 4th, 2022)

A bugfix release that solves the following issues:

* Missing feature values are now dealt with correctly when using feature binning.
* A rare issue that may cause segmentation faults when using instance sampling has been fixed.

## Version 0.8.0 (Jan. 31, 2022)

***This release comes with changes to the command line API. For an updated overview of the available parameters, please refer to the [documentation](https://mlrl-boomer.readthedocs.io).***

A major update to the BOOMER algorithm that introduces the following changes:

* The programmatic C++ API was redesigned for a more convenient configuration of algorithms. This does also drastically reduce the amount of wrapper code that is necessary to access the API from other programming languages and therefore facilitates the support of additional languages in the future.
* An issue that may cause segmentation faults when using stratified sampling methods for the creation of holdout sets has been fixed.
* Pre-built packages for Windows systems are now available at [PyPI](https://pypi.org/project/mlrl-boomer/).
* Pre-built packages for Linux aarch64 systems are now provided.

## Version 0.7.1 (Dec. 15, 2021)

A bugfix release that solves the following issues:

* Fixes an issue preventing the use of dense representations of ground truth label matrices that was introduced in version 0.7.0.
* Pre-built packages for MacOS systems are now available at [PyPI](https://pypi.org/project/mlrl-boomer/).
* Linux and MacOS packages for Python 3.10 are now provided.

## Version 0.7.0 (Dec. 5, 2021)

A major update to the BOOMER algorithm that introduces the following changes:

* L1 regularization can now be used.
* A more space-efficient data structure is now used for the sparse representation of binary predictions.
* The Python API does now allow to access the rules in a model in a programmatic way.
* It is now possible to output certain characteristics of training datasets and rule models.
* Pre-built packages for the Linux platform are now available at [PyPI](https://pypi.org/project/mlrl-boomer/).
* The [documentation](https://mlrl-boomer.readthedocs.io) has vastly been improved.

## Version 0.6.2 (Oct 4, 2021)

A bugfix release that solves the following issues:

* Fixes a segmentation fault when a sparse feature matrix should be used for prediction that was introduced in version 0.6.0.

## Version 0.6.1 (Sep 30, 2021)

A bugfix release that solves the following issues:

* Fixes a mathematical problem when calculating the quality of potential single-label rules that was introduced in version 0.6.0.

## Version 0.6.0 (Sep 6, 2021)

***This release comes with changes to the command line API. For brevity and consistency, some parameters and/or their values have been renamed. Moreover, some parameters have been updated to use more reasonable default values. For an updated overview of the available parameters, please refer to the [documentation](https://mlrl-boomer.readthedocs.io).***

A major update to the BOOMER algorithm that introduces the following changes:

* The parameter `--instance-sampling` does now allow to use stratified sampling (`stratified-label-wise` and `stratified-example-wise`).
* The parameter `--holdout` does now allow to use stratified sampling (`stratified-label-wise` and `stratified-example-wise`).
* The parameter `--recalculate-predictions` does now allow to specify whether the predictions of rules should be recalculated on the entire training data, if instance sampling is used.
* An additional parameter (`--prediction-format`) that allows to specify whether predictions should be stored using dense or sparse matrices has been added. 
* The code for the construction of rule heads has been reworked, resulting in minor performance improvements.
* The unnecessary calculation of Hessians is now avoided when used single-label rules for the minimization of a non-decomposable loss function, resulting in a significant performance improvement.
* A programmatic C++ API for configuring algorithms, including the validation of parameters, is now provided.
* A documentation is now available [online](https://mlrl-boomer.readthedocs.io).

## Version 0.5.0 (Jun 27, 2021)

A major update to the BOOMER algorithm that introduces the following changes:

* Gradient-based label binning (GBLB) can be used to assign labels to a predefined number of bins.

## Version 0.4.0 (Mar 31, 2021)

A major update to the BOOMER algorithm that introduces the following changes:

* Large parts of the code have been refactored, and the core algorithm has been migrated to C++ entirely. It is now built and compiled using Meson and Ninja, which results in drastically reduced compile times.
* The (label- and example-wise) logistic loss functions have been rewritten to better prevent numerical problems.
* Approximate methods for evaluating potential conditions of rules, based on unsupervised binning methods (currently equal-width- and equal-frequency-binning), have been added.
* The parameter `--predictor` does now allow using different algorithms for prediction (`label-wise` or `example-wise`).
* An early stopping mechanism has been added, which allows to stop the induction of rules as soon as the quality of the model does not improve on a holdout set.    
* Multi-threading can be used to parallelize the prediction for different examples across multiple CPU cores.
* Multi-threading can be used to parallelize the calculation of gradients and Hessians for different examples across multiple CPU cores.
* Probability estimates can be predicted when using the loss function `label-wise-logistic-loss`.
* The algorithm does now support data sets with missing feature values.
* The loss function `label-wise-squared-hinge-loss` has been added. 
* Experiments using single-label data sets are now supported out of the box.

## Version 0.3.0 (Sep 14, 2020)

A major update to the BOOMER algorithm that features the following changes:

* Large parts of the code (loss functions, calculation of gradients/Hessians, calculation of predictions/quality scores) have been refactored and rewritten in C++. This comes with a constant speed-up of training times.
* Multi-threading can be used to parallelize the evaluation of a rule's potential refinements across multiple CPU cores.
* Sparse ground truth label matrices can now be used for training, which may reduce the memory footprint in case of large data sets.
* Additional parameters (`feature-format` and `label-format`) that allow to specify the preferred format of the feature and label matrix have been added.

## Version 0.2.0 (Jun 28, 2020)

A major update to the BOOMER algorithm that features the following changes:

* Includes many refactorings and quality of live improvements. Code that is not directly related with the algorithm, such as the implementation of baselines, has been removed.
* The algorithm is now able to natively handle nominal features without the need for pre-processing techniques such as one-hot encoding.
* Sparse feature matrices can now be used for training and prediction, which reduces the memory footprint and results in a significant speed-up of training times on some data sets.
* Additional hyper-parameters (`min_coverage`, `max_conditions` and `max_head_refinements`) that provide fine-grained control over the specificity/generality of rules have been added.

## Version 0.1.0 (Jun 22, 2020)

The first version of the BOOMER algorithm used in the following publication:

*Michael Rapp, Eneldo Loza Mencía, Johannes Fürnkranz and Eyke Hüllermeier. Gradient-based Label Binning in Multi-label Classification. In: Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), 2021, Springer.*

This version supports the following features to learn an ensemble of boosted classification rules:

* Different label-wise or example-wise loss functions can be minimized during training (optionally using L2 regularization).
* The rules may predict for a single label, or for all labels (which enables to model local label dependencies).
* When learning a new rule, random samples of the training examples, features or labels may be used, including different techniques such as sampling with or without replacement.
* The impact of individual rules on the ensemble can be controlled using shrinkage.
* The conditions of a recently induced rule can be pruned based on a hold-out set.
* The algorithm currently only supports numerical or ordinal features. Nominal features can be handled by using one-hot encoding.
