# README

This project provides a scikit-learn implementation of "BOOMER" - an algorithm for learning gradient boosted multi-label classification rules.

The algorithm was first published in the following [paper](https://link.springer.com/chapter/10.1007/978-3-030-67664-3_8). A preprint version is publicly available [here](https://arxiv.org/pdf/2006.13346.pdf).

*Rapp M., Loza Mencía E., Fürnkranz J., Nguyen VL., Hüllermeier E. (2020) Learning Gradient Boosted Multi-label Classification Rules. In: Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2020. Lecture Notes in Computer Science, pp. 124-140, vol 12459. Springer, Cham*

Gradient-based label binning (GBLB), which is an extension to the original algorithm, was proposed in the following paper. A preprint version is available [here](https://arxiv.org/pdf/2106.11690.pdf).

*Rapp M., Loza Mencía E., Fürnkranz J., Hüllermeier E. (2021) Gradient-based Label Binning in Multi-label Classification. In: Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2021. Lecture Notes in Computer Science, vol 12977. Springer, Cham*

If you use the algorithm in a scientific publication, we would appreciate citations to the mentioned papers.

## Features

The algorithm that is provided by this project currently supports the following core functionalities to learn an ensemble of boosted classification rules:

* Different label-wise or example-wise loss functions can be minimized during training (optionally using L2 regularization).
* The rules may predict for a single label or for all labels (which enables to model local label dependencies).
* When learning a new rule, random samples of the training examples, features or labels may be used (including different techniques such as sampling with or without replacement).
* The impact of individual rules on the ensemble can be controlled using shrinkage.
* Hyper-parameters that provide fine-grained control over the specificity/generality of rules are available.
* The conditions of rules can be pruned based on a hold-out set.
* The algorithm can natively handle numerical, ordinal and nominal features (without the need for pre-processing techniques such as one-hot encoding).
* The algorithm is able to deal with missing feature values, i.e., occurrences of NaN in the feature matrix.
* Different strategies for prediction, which can be tailored to the used loss function, are available.

In addition, the following features that may speed up training or reduce the memory footprint are currently implemented:

* Approximate methods for evaluating potential conditions of rules, based on unsupervised binning methods, can be used.
* Gradient-based label binning (GBLB) can be used to assign the available labels to a limited number of bins. The use of label binning may speed up training significantly when using rules that predict for multiple labels to minimize a non-decomposable loss function.
* Dense or sparse feature matrices can be used for training and prediction. The use of sparse matrices may speed up training significantly on some data sets.
* Dense or sparse label matrices can be used for training. The use of sparse matrices may reduce the memory footprint in case of large data sets.
* Dense or sparse matrices can be used to store predictions. The use of sparse matrices may reduce the memory footprint in case of large data sets.
* Multi-threading can be used to parallelize the evaluation of a rule's potential refinements across multiple CPU cores. 

## Project structure

```
|-- data                                Contains several benchmark data sets
    |-- ...
|-- data-synthetic                      Contains several synthetic data sets
    |-- ...
|-- cpp                                 Contains the implementation of core algorithms in C++
    |-- subprojects
        |-- common                      Contains implementations that all algorithms have in common
        |-- boosting                    Contains implementations of boosting algorithms
        |-- seco                        Contains implementations of separate-and-conquer algorithms
    |-- ...
|-- python                              Contains Python code for running experiments using different algorithms
    |-- mlrl
        |-- common                      Contains Python code that is needed to run any kind of algorithms
            |-- cython                  Contains commonly used Cython wrappers
            |-- ...
        |-- boosting                    Contains Python code for running boosting algorithms
            |-- cython                  Contains boosting-specific Cython wrappers
            |-- ...
        |-- seco                        Contains Python code for running separate-and-conquer algorithms
            |-- cython                  Contains separate-and-conquer-specific Cython wrappers
            |-- ...
        |-- testbed                     Contains useful functionality for running experiments, e.g., for cross validation, writing of output files, etc.
            |-- ...
    |-- main_boomer.py                  Can be used to start an experiment using the BOOMER algorithm
    |-- main_seco.py                    Can be used to start an experiment using the separate-and-conquer algorithm
    |-- main_generate_synthetic_data.py Can be used to generate synthetic data sets
    |-- ...
|-- Makefile                            Makefile for compilation
|-- ...
```

## Project setup

The algorithm provided by this project is implemented in C++. In addition, a Python wrapper that implements the scikit-learn API is available. To be able to integrate the underlying C++ implementation with Python, [Cython](https://cython.org) is used.

The C++ implementation, as well as the Cython wrappers, must be compiled in order to be able to run the provided algorithm. To facilitate compilation, this project comes with a Makefile that automatically executes the necessary steps.

At first, a virtual Python environment can be created via the following command:
```
make venv
```

As a prerequisite, Python 3.7 (or a more recent version) must be available on the host system. All compile-time dependencies (`numpy`, `scipy`, `Cython`, `meson` and `ninja`) that are required for building the project will automatically be installed into the virtual environment. As a result of executing the above command, a subdirectory `venv` should have been created within the project's root directory.

Afterwards, the compilation can be started by executing the following command:
```
make compile
```

Finally, the library must be installed into the virtual environment, together with all of its runtime dependencies (e.g. `scikit-learn`, a full list can be found in `setup.py`). For this purpose, the project's Makefile provides the following command:

```
make install
```

*Whenever any C++ or Cython source files have been modified, they must be recompiled by running the command `make compile` again! If compilation files do already exist, only the modified files will be recompiled.*

**Cleanup:** To get rid of any compilation files, as well as of the virtual environment, the following command can be used:
```
make clean
``` 

For more fine-grained control, the command `make clean_venv` (for deleting the virtual environment) or `make clean_compile` (for deleting the compiled files) can be used. If only the compiled Cython files should be removed, the command `make clean_cython` can be used. Accordingly, the command `make clean_cpp` removes the compiled C++ files.

**Syntax highlighting in PyCharm:** Unfortunately, the Community Edition of PyCharm does not come with Cython support. To enable proper syntax highlighting of Cython code, the file `settings.zip` in the project's root directory can be imported via `File -> Import Settings`.

## Running experiments

The file `main_boomer.py` allows to run experiments on a specific data set using different configurations of the learning algorithm. The implementation takes care of writing the experimental results into `.csv` files and the learned model can (optionally) be stored on disk to reuse it later. 

In order to run an experiment, the following command line arguments must be provided (most of them are optional):

| Parameter                     | Optional? | Default             | Description                                                                                                                                                                                                                                                                                                                       |
|-------------------------------|-----------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--data-dir`                  | No        | None                | The path of the directory where the data set files are located.                                                                                                                                                                                                                                                                   |
| `--dataset`                   | No        | None                | The name of the data set file files without suffix.                                                                                                                                                                                                                                                                               |
| `--folds`                     | Yes       | 1                   | The total number of folds to be used for cross validation. Must be greater than 1 or 1, if no cross validation should be used.                                                                                                                                                                                                    |
| `--current-fold`              | Yes       | 0                   | The cross validation fold to be performed. Must be in [1, --folds] or 0, if all folds should be performed. This parameter is ignored if --folds is set to 1.                                                                                                                                                                      |
| `--evaluate-training-data`    | Yes       | false               | Whether the models should not only be evaluated on the test data, but also on the training data. Must be `true` or `false`.                                                                                                                                                                                                       |
| `--one-hot-encoding`          | Yes       | false               | Whether one-hot-encoding should be used to encode nominal attributes or not. Must be `true` or `false`                                                                                                                                                                                                                            |
| `--feature-format`            | Yes       | auto                | The format to be used for the representation of the feature matrix. Must be `sparse`, `dense` or `auto`.                                                                                                                                                                                                                          |
| `--label-format`              | Yes       | auto                | The format to be used for the representation of the label matrix. Must be `sparse`, `dense` or `auto`.                                                                                                                                                                                                                            |
| `--prediction-format`         | Yes       | auto                | The format to be used for the representation of predicted labels. Must be `sparse`, `dense` or `auto`.                                                                                                                                                                                                                            |
| `--model-dir`                 | Yes       | None                | The path of the directory where models should be stored.                                                                                                                                                                                                                                                                          |
| `--parameter-dir`             | Yes       | None                | The path of the directory where configuration files, which specify the parameters to be used by the algorithm, are located.                                                                                                                                                                                                       |
| `--output-dir`                | Yes       | None                | The path of the directory where experimental results should be saved.                                                                                                                                                                                                                                                             |
| `--store-predictions`         | Yes       | false               | Whether the predictions for individual examples and labels should be written into output files or not. Must be `true` or `false`. Does only have an effect if the parameter --output-dir is specified.                                                                                                                            |
| `--print-rules`               | Yes       | false               | Whether the induced rules should be printed on the console or not. Must be `true` or `false`.                                                                                                                                                                                                                                     |
| `--store-rules`               | Yes       | false               | Whether the induced rules should be written into a text file or not. Must be `true` or `false`. Does only have an effect if the parameter --output-dir is specified.                                                                                                                                                              |
| `--log-level`                 | Yes       | info                | The log level to be used. Must be `debug`, `info`, `warn`, `warning`, `error`, `critical`, `fatal` or `notset`.                                                                                                                                                                                                                   |
| `--print-options`             | Yes       | None                | Additional options to be taken into account when writing rules on the console or into an output file. Does only have an effect if the parameter --print-rules or --store-rules is set to `true`. For a list of the available options refer to the documentation.                                                                  |
| `--random-state`              | Yes       | 1                   | The seed to the be used by random number generators. Must be at least 1.                                                                                                                                                                                                                                                          |
| `--max-rules`                 | Yes       | 1000                | The number of rules to be induced. Must be at least 1 or 0, if the number of rules should not be restricted.                                                                                                                                                                                                                      |
| `--default-rule`              | Yes       | true                | Whether the first rule should be a default rule or not. Must be `true` or `false`.                                                                                                                                                                                                                                                |
| `--time-limit`                | Yes       | 0                   | The duration in seconds after which the induction of rules should be canceled. Must be at least 1 or 0, if no time limit should be set.                                                                                                                                                                                           |
| `--label-sampling`            | Yes       | None                | The name of the strategy to be used for label sampling. Must be `without-replacement` or `None`. For additional options refer to the documentation.                                                                                                                                                                               |                                                       
| `--feature-sampling`          | Yes       | without-replacement | The name of the strategy to be used for feature sampling. Must be `without-replacement` or `None`. For additional options refer to the documentation.                                                                                                                                                                             |
| `--instance-sampling`         | Yes       | None                | The name of the strategy to be used for instance sampling. Must be `with-replacement`, `without-replacement`, `stratified-label-wise`, `stratified-example-wise` or `None`, if no instance sampling should be used. For additional options refer to the documentation.                                                            |
| `--recalculate-predictions`   | Yes       | true                | Whether the predictions of rules should be recalculated on the entire training data, if the parameter --instance-sampling is not set to `None`, or not. Must be `true` or `false`.                                                                                                                                                |
| `--holdout`                   | Yes       | None                | The name of the strategy to be used for creating a holdout set. Must be `random`, `stratified-label-wise`, `stratified-example-wise` or `None`, if no holdout set should be created. For additional options refer to the documentation.                                                                                           |
| `--early-stopping`            | Yes       | None                | The name of the strategy to be used for early stopping. Must be `loss` or `None`, if no early stopping should be used. Does only have an effect if the parameter --holdout is not set to `None`. For additional options refer to the documentation.                                                                               |                                                                                
| `--feature-binning`           | Yes       | None                | The name of the strategy to be used for feature binning. Must be `equal-width`, `equal-frequency` or `None`, if no feature binning should be used. For additional options refer to the documentation.                                                                                                                             |
| `--label-binning`             | Yes       | auto                | The name of the strategy to be used for gradient-based label binning (GBLB). Must be `auto`, `equal-width` or `None`, if no label binning should be used. If set to `auto`, the most suitable strategy is chosen automatically based on the parameters --loss and --head-type. For additional options refer to the documentation. |
| `--pruning`                   | Yes       | None                | The name of the strategy to be used for pruning rules. Must be `irep` or `None`, if no pruning should be used. Does only have an effect if the parameter --instance-sampling is not set to `None`.                                                                                                                                |
| `--min-coverage`              | Yes       | 1                   | The minimum number of training examples that must be covered by a rule. Must be at least 1.                                                                                                                                                                                                                                       |
| `--max-conditions`            | Yes       | 0                   | The maximum number of conditions to be included in a rule's body. Must be at least 1 or 0, if the number of conditions should not be restricted.                                                                                                                                                                                  |
| `--max-head-refinements`      | Yes       | 1                   | The maximum number of times the head of a rule may be refined. Must be at least 1 or 0, if the number of refinements should not be restricted.                                                                                                                                                                                    |
| `--head-type`                 | Yes       | auto                | The type of the rule heads that should be used. Must be `single-label`, `complete` or `auto`. If set to `auto`, the most suitable type is chosen automatically based on the parameter --loss.                                                                                                                                     |
| `--shrinkage`                 | Yes       | 0.3                 | The shrinkage parameter, a.k.a. the learning rate, to be used. Must be in (0, 1].                                                                                                                                                                                                                                                 |
| `--loss`                      | Yes       | logistic-label-wise | The name of the loss function to be minimized during training. Must be `squared-error-label-wise`, `hinge-label-wise`, `logistic-label-wise` or `logistic-example-wise`.                                                                                                                                                          |
| `--predictor`                 | Yes       | auto                | The name of the strategy to be used for making predictions. Must be `label-wise`, `example-wise` or `auto`. If set to `auto`, the most suitable strategy is chosen automatically based on the parameter --loss.                                                                                                                   |
| `--l2-regularization-weight`  | Yes       | 1.0                 | The weight of the L2 regularization. Must be at least `0`.                                                                                                                                                                                                                                                                        |
| `--parallel-rule-refinement`  | Yes       | 1                   | The number of threads to be used to search for potential refinements of rules in parallel. Must be at least 1 or 0, if the number of cores that are available on the machine should be used.                                                                                                                                      |
| `--parallel-statistic-update` | Yes       | 1                   | The number of threads to be used to calculate gradients and Hessians for different examples in parallel. Must be at least 1 or 0, if the number of cores that are available on the machine should be used.                                                                                                                        |
| `--parallel-prediction`       | Yes       | 1                   | The number of threads to be used to make predictions for different examples in parallel. Must be at least 1 or 0, if the number of cores that are available on the machine should be used.                                                                                                                                        |

In the following, the command for running an experiment using an exemplary configuration can be seen. It uses a virtual environment as discussed in section "Project setup". 

```
venv/bin/python3 python/main_boomer.py --data-dir /path/to/data --output-dir /path/to/results/emotions --model-dir /path/to/models/emotions --dataset emotions --folds 10 --max-rules 1000 --instance-sampling with-replacement --feature-sampling without-replacement --loss logistic-label-wise --shrinkage 0.3 --pruning None --head-type single-label
```
