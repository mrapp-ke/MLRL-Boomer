---
tocdepth: 2
---

(release-notes)=

# Release Notes

## Version 0.11.2 (Jan. 23rd, 2025)

A bugfix release that comes with the following changes.

### Fixes

- When building macOS packages, we do now manually compile OpenMP to ensure that the bundled library matches the target platform.

### Quality-of-Life Improvements

- The build system now uses a lightweight custom implementation instead of [SCons](https://scons.org/) and is better modularized to avoid unnecessary runs of Continuous Integration jobs when only certain parts of it are modified.
- Releases are now automated via Continuous Integration, including the update of the project's changelog.
- The presentation of algorithmic parameters in the documentation has been improved.
- Continuous Integration is now used to automatically update outdated GitHub Actions, outdated Python dependencies, and the Doxygen configuration on a regular basis.
- Continuous Integration is now used to check the documentation for broken links and spelling mistakes.
- The syntax of TOML files is now validated by applying [taplo](https://github.com/tamasfe/taplo) via Continuous Integration.

## Version 0.11.1 (Sep. 24th, 2024)

A bugfix release that fixes the following issues:

- A crash has been fixed that could occur when using the command line argument `--print-rules` or `--store-rules` with a dataset that contains nominal features.

## Version 0.11.0 (Aug. 10th, 2024)

A major update to the BOOMER algorithm that introduces the following changes.

```{warning}
This release comes with several API changes. For an updated overview of the available parameters and command line arguments, please refer to the [documentation](https://mlrl-boomer.readthedocs.io/en/0.11.0/).
```

### Algorithmic Enhancements

- **The BOOMER algorithm can be used for solving regression problems**, including single- and multi-output regression problems.

### Additions to the Command Line API

- **Custom algorithms can now be easily integrated** with the command line API due to the ability to dynamically load code from your own Python modules or source files, as illustrated [here](https://mlrl-boomer.readthedocs.io/en/0.11.0/user_guide/testbed/runnables.html)
- **The value to be used for sparse elements in the feature matrix can be specified** via the argument `--sparse-feature-value`.

### API Changes

- The Python module or source file providing an integration with the machine learning algorithm to be used by the command line API must now be specified as described [here](https://mlrl-boomer.readthedocs.io/en/0.11.0/user_guide/testbed/arguments.html#basic-usage).
- Several parameters and their values have been renamed to better reflect the scope of the project, which now includes multi-output regression problems. For an up-to-date list of parameters, please refer to the [documentation](https://mlrl-boomer.readthedocs.io/en/0.11.0/).
- Rules with complete heads are now learned by default when using a decomposable loss function and a dense format for storing statistics.

## Version 0.10.2 (Aug. 9th, 2024)

A bugfix release that fixes the following issues.

- A rare issue that may result in rules with empty heads being learned when using the argument `--head-type partial-dynamic` has been fixed.

## Version 0.10.1 (Aug. 1st, 2024)

A bugfix release that comes with the following changes.

- If the sparse value of a feature matrix is provided to the Python API, it is now properly taken into account when converting into a dense matrix.
- The C++ code is now checked for common issues by applying `cpplint` via Continuous Integration.
- The styling of YAML files is now verified by applying `yamlfix` via Continuous Integration.

## Version 0.10.0 (May 5th, 2024)

A major update to the BOOMER algorithm that introduces the following changes.

```{warning}
This release comes with several API changes. For an updated overview of the available parameters and command line arguments, please refer to the [documentation](https://mlrl-boomer.readthedocs.io/en/0.10.0/).
```

### Algorithmic Enhancements

- **The project does now provide a Separate-and-Conquer (SeCo) algorithm** based on traditional rule learning techniques that are particularly well-suited for learning interpretable models.
- **Space-efficient data structures are now used for storing feature values**, depending on whether the feature is numerical, ordinal, nominal, or binary. This also enables to use optimized code paths for dealing with these different types of features.
- **The implementation of feature binning has been reworked** in a way that avoids redundant code and results in a reduction of training times due to the use of the data structures mentioned above.
- **The value to be used for sparse elements of a feature matrix** can now be specified via the C++ or Python API.
- **Nominal and ordinal feature values are now represented as integers** to avoid issues due to limited floating point precision.
- **Safe comparisons of floating point values** are now used to avoid issues due to limited floating point precision.
- **Fundamental data structures for vectors and matrices have been reworked** to ease reusing existing functionality and avoiding redundant code.

### Additions to the Command Line API

- **Information about the program can now be printed** via the argument `-v` or `--version`.
- **Data characteristics do now include the number of ordinal features** when printed on the console or written to a file via the command line argument `--print-data-characteristics` or `--store-data-characteristics`.

### Bugfixes

- An issue has been fixed that caused the number of numerical and nominal features to be swapped when using the command line arguments `--print-data-characteristics` or `--store-data-characteristics`.
- The correct directory is now used for loading and saving parameter settings when using the command line arguments `--parameter-dir` and `--store-parameters`.

### API Changes

- The option `num_threads` of the parameters `--parallel-rule-refinement`, `--parallel-statistic-update` and `--parallel-prediction` has been renamed to `num_preferred_threads`.

### Quality-of-Life Improvements

- The documentation has been updated to a more modern theme supporting light and dark theme variants.
- A build option that allows disabling multi-threading support via OpenMP at compile-time has been added.
- The groundwork for GPU support was laid. It can be disabled at compile-time via a build option.
- Added support for unit testing the project's C++ code. Compilation of the tests can be disabled via a build option.
- The Python code is now checked for common issues by applying `pylint` via Continuous Integration.
- The Makefile has been replaced with wrapper scripts triggering a [SCons](https://scons.org/) build.
- Development versions of wheel packages are now regularly built via Continuous Integration, uploaded as artifacts, and published on [Test-PyPI](https://test.pypi.org/).
- Continuous integration is now used to maintain separate branches for major, feature, and bugfix releases and keep them up-to-date.
- The runtime of Continuous Integration jobs has been optimized by running individual steps only if necessary, caching files across subsequent runs, and making use of parallelization.
- When tests are run via Continuous Integration, a summary of the test results is now added to pull requests and GitHub workflows.
- Markdown files are now used for writing the documentation.
- A consistent style is now enforced for Markdown files by applying the tool `mdformat` via Continuous Integration.
- C++ 17 or newer is now required for compiling the project.

## Version 0.9.0 (Jul. 2nd, 2023)

A major update to the BOOMER algorithm that introduces the following changes.

```{warning}
This release comes with several API changes. For an updated overview of the available parameters and command line arguments, please refer to the [documentation](https://mlrl-boomer.readthedocs.io/en/0.9.0/).
```

### Algorithmic Enhancements

- **Sparse matrices can now be used to store gradients and Hessians** if supported by the loss function. The desired behavior can be specified via a new parameter `--statistic-format`.
- **Rules with partial heads can now be learned** by setting the parameter `--head-type` to the value `partial-fixed`, if the number of predicted labels should be predefined, or `partial-dynamic`, if the subset of predicted labels should be determined dynamically.
- **A beam search can now be used** for the induction of individual rules by setting the parameter `--rule-induction` to the value `top-down-beam-search`.
- **Variants of the squared error loss and squared hinge loss**, which take all labels of an example into account at the same time, can now be used by setting the parameter `--loss` to the value `squared-error-example-wise` or `squared-hinge-example-wise`.
- **Probability estimates can be obtained for each label independently or via marginalization** over the label vectors encountered in the training data by setting the new parameter `--probability-predictor` to the value `label-wise` or `marginalized`.
- **Predictions that maximize the example-wise F1-measure can now be obtained** by setting the parameter `--classification-predictor` to the value `gfm`.
- **Binary predictions can now be derived from probability estimates** by specifying the new option `based_on_probabilities`.
- **Isotonic regression models can now be used** to calibrate marginal and joint probabilities predicted by a model via the new parameters `--marginal-probability-calibration` and `--joint-probability-calibration`.
- **The rules in a previously learned model can now be post-optimized** by reconstructing each one of them in the context of the other rules via the new parameter `--sequential-post-optimization`.
- **Early stopping or post-pruning can now be used** by setting the new parameter `--global-pruning` to the value `pre-pruning` or `post-pruning`.
- **Single labels can now be sampled in a round-robin fashion** by setting the parameter `--feature-sampling` to the new value `round-robin`.
- **A fixed number of trailing features can now be retained** when the parameter `--feature-sampling` is set to the value `without-replacement` by specifying the option `num_retained`.

### Additions to the Command Line API

- **Data sets in the MEKA format are now supported.**
- **Certain characteristics of binary predictions can be printed or written to output files** via the new arguments `--print-prediction-characteristics` and `--store-prediction-characteristics`.
- **Unique label vectors contained in the training data can be printed or written to output files** via the new arguments `--print-label-vectors` and `--store-label-vectors`.
- **Models for the calibration of marginal or joint probabilities can be printed or written to output files** via the new arguments `--print-marginal-probability-calibration-model`, `--store-marginal-probability-calibration-model`, `--print-joint-probability-calibration-model` and `--store-joint-probability-calibration-model`.
- **Models can now be evaluated repeatedly, using a subset of their rules with increasing size,** by specifying the argument `--incremental-prediction`.
- **More control of how data is split into training and test sets** is now provided by the argument `--data-split` that replaces the arguments `--folds` and `--current-fold`.
- **Binary labels, scores, or probabilities can now be predicted,** depending on the value of the new argument `--prediction-type`, which can be set to the values `binary`, `scores`, or `probabilities`.
- **Individual evaluation measures can now be enabled or disabled** via additional options that have been added to the arguments `--print-evaluation` and `--store-evaluation`.
- **The presentation of values printed on the console has vastly been improved.** In addition, options for controlling the presentation of values to be printed or written to output files have been added to various command line arguments.

### Bugfixes

- The behavior of the parameter `--label-format` has been fixed when set to the value `auto`.
- The behavior of the parameters `--holdout` and `--instance-sampling` has been fixed when set to the value `stratified-label-wise`.
- The behavior of the parameter `--binary-predictor` has been fixed when set to the value `example-wise` and using a model that has been loaded from disk.
- Rules are now guaranteed to not cover more examples than specified via the option `min_coverage`. The option is now also taken into account when using feature binning. Alternatively, the minimum coverage of rules can now also be specified as a fraction via the option `min_support`.

### API Changes

- The parameter `--early-stopping` has been replaced with a new parameter `--global-pruning`.
- The parameter `--pruning` has been renamed to `--rule-pruning`.
- The parameter `--classification-predictor` has been renamed to `--binary-predictor`.
- The command line argument `--predict-probabilities` has been replaced with a new argument `--prediction-type`.
- The command line argument `--predicted-label-format` has been renamed to `--prediction-format`.

### Quality-of-Life Improvements

- Continuous integration is now used to test the most common functionalities of the BOOMER algorithm and the corresponding command line API.
- Successful generation of the documentation is now tested via Continuous Integration.
- Style definitions for Python and C++ code are now enforced by applying the tools `clang-format`, `yapf`, and `isort` via Continuous Integration.

## Version 0.8.2 (Apr. 11th, 2022)

A bugfix release that solves the following issues:

- Fixed prebuilt packages available at [PyPI](https://pypi.org/project/mlrl-boomer/).
- Fixed output of nominal values when using the option `--print-rules true`.

## Version 0.8.1 (Mar. 4th, 2022)

A bugfix release that solves the following issues:

- Missing feature values are now dealt with correctly when using feature binning.
- A rare issue that may cause segmentation faults when using instance sampling has been fixed.

## Version 0.8.0 (Jan. 31, 2022)

A major update to the BOOMER algorithm that introduces the following changes.

```{warning}
This release comes with changes to the command line API. For an updated overview of the available parameters, please refer to the [documentation](https://mlrl-boomer.readthedocs.io/en/0.8.0/).
```

- The programmatic C++ API was redesigned for a more convenient configuration of algorithms. This does also drastically reduce the amount of wrapper code that is necessary to access the API from other programming languages and therefore facilitates the support of additional languages in the future.
- An issue that may cause segmentation faults when using stratified sampling methods for the creation of holdout sets has been fixed.
- Pre-built packages for Windows systems are now available at [PyPI](https://pypi.org/project/mlrl-boomer/).
- Pre-built packages for Linux aarch64 systems are now provided.

## Version 0.7.1 (Dec. 15, 2021)

A bugfix release that solves the following issues:

- Fixes an issue preventing the use of dense representations of ground truth label matrices that was introduced in version 0.7.0.
- Pre-built packages for macOS systems are now available at [PyPI](https://pypi.org/project/mlrl-boomer/).
- Linux and macOS packages for Python 3.10 are now provided.

## Version 0.7.0 (Dec. 5, 2021)

A major update to the BOOMER algorithm that introduces the following changes:

- L1 regularization can now be used.
- A more space-efficient data structure is now used for the sparse representation of binary predictions.
- The Python API does now allow to access the rules in a model in a programmatic way.
- It is now possible to output certain characteristics of training datasets and rule models.
- Pre-built packages for the Linux platform are now available at [PyPI](https://pypi.org/project/mlrl-boomer/).
- The [documentation](https://mlrl-boomer.readthedocs.io) has vastly been improved.

## Version 0.6.2 (Oct. 4, 2021)

A bugfix release that solves the following issues:

- Fixes a segmentation fault when a sparse feature matrix should be used for prediction that was introduced in version 0.6.0.

## Version 0.6.1 (Sep. 30, 2021)

A bugfix release that solves the following issues:

- Fixes a mathematical problem when calculating the quality of potential single-label rules that was introduced in version 0.6.0.

## Version 0.6.0 (Sep. 6, 2021)

A major update to the BOOMER algorithm that introduces the following changes.

```{warning}
This release comes with changes to the command line API. For brevity and consistency, some parameters and/or their values have been renamed. Moreover, some parameters have been updated to use more reasonable default values. For an updated overview of the available parameters, please refer to the [documentation](https://mlrl-boomer.readthedocs.io/en/0.6.0/).
```

- The parameter `--instance-sampling` does now allow to use stratified sampling (`stratified-label-wise` and `stratified-example-wise`).
- The parameter `--holdout` does now allow to use stratified sampling (`stratified-label-wise` and `stratified-example-wise`).
- The parameter `--recalculate-predictions` does now allow to specify whether the predictions of rules should be recalculated on the entire training data, if instance sampling is used.
- An additional parameter (`--prediction-format`) that allows to specify whether predictions should be stored using dense or sparse matrices has been added.
- The code for the construction of rule heads has been reworked, resulting in minor performance improvements.
- The unnecessary calculation of Hessians is now avoided when used single-label rules for the minimization of a non-decomposable loss function, resulting in a significant performance improvement.
- A programmatic C++ API for configuring algorithms, including the validation of parameters, is now provided.
- A documentation is now available [online](https://mlrl-boomer.readthedocs.io).

## Version 0.5.0 (Jun. 27, 2021)

A major update to the BOOMER algorithm that introduces the following changes:

- Gradient-based label binning (GBLB) can be used to assign labels to a predefined number of bins.

## Version 0.4.0 (Mar. 31, 2021)

A major update to the BOOMER algorithm that introduces the following changes:

- Large parts of the code have been refactored, and the core algorithm has been migrated to C++ entirely. It is now built and compiled using Meson and Ninja, which results in drastically reduced compile times.
- The (label- and example-wise) logistic loss functions have been rewritten to better prevent numerical problems.
- Approximate methods for evaluating potential conditions of rules, based on unsupervised binning methods (currently equal-width- and equal-frequency-binning), have been added.
- The parameter `--predictor` does now allow using different algorithms for prediction (`label-wise` or `example-wise`).
- An early stopping mechanism has been added, which allows to stop the induction of rules as soon as the quality of the model does not improve on a holdout set.
- Multi-threading can be used to parallelize the prediction for different examples across multiple CPU cores.
- Multi-threading can be used to parallelize the calculation of gradients and Hessians for different examples across multiple CPU cores.
- Probability estimates can be predicted when using the loss function `label-wise-logistic-loss`.
- The algorithm does now support data sets with missing feature values.
- The loss function `label-wise-squared-hinge-loss` has been added.
- Experiments using single-label data sets are now supported out of the box.

## Version 0.3.0 (Sep. 14, 2020)

A major update to the BOOMER algorithm that features the following changes:

- Large parts of the code (loss functions, calculation of gradients/Hessians, calculation of predictions/quality scores) have been refactored and rewritten in C++. This comes with a constant speed-up of training times.
- Multi-threading can be used to parallelize the evaluation of a rule's potential refinements across multiple CPU cores.
- Sparse ground truth label matrices can now be used for training, which may reduce the memory footprint in case of large data sets.
- Additional parameters (`feature-format` and `label-format`) that allow to specify the preferred format of the feature and label matrix have been added.

## Version 0.2.0 (Jun. 28, 2020)

A major update to the BOOMER algorithm that features the following changes:

- Includes many refactorings and quality of live improvements. Code that is not directly related with the algorithm, such as the implementation of baselines, has been removed.
- The algorithm is now able to natively handle nominal features without the need for pre-processing techniques such as one-hot encoding.
- Sparse feature matrices can now be used for training and prediction, which reduces the memory footprint and results in a significant speed-up of training times on some data sets.
- Additional hyperparameters (`min_coverage`, `max_conditions` and `max_head_refinements`) that provide fine-grained control over the specificity/generality of rules have been added.

## Version 0.1.0 (Jun. 22, 2020)

The first version of the BOOMER algorithm used in the following publication:

*Michael Rapp, Eneldo Loza Mencía, Johannes Fürnkranz and Eyke Hüllermeier. Gradient-based Label Binning in Multi-label Classification. In: Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), 2021, Springer.*

This version supports the following features to learn an ensemble of boosted classification rules:

- Different label-wise or example-wise loss functions can be minimized during training (optionally using L2 regularization).
- The rules may predict for a single label, or for all labels (which enables to model local label dependencies).
- When learning a new rule, random samples of the training examples, features or labels may be used, including different techniques such as sampling with or without replacement.
- The impact of individual rules on the ensemble can be controlled using shrinkage.
- The conditions of a recently induced rule can be pruned based on a hold-out set.
- The algorithm currently only supports numerical or ordinal features. Nominal features can be handled by using one-hot encoding.
