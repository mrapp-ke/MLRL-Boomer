# README

This project provides a scikit-learn implementation of "BOOMER" - an algorithm for learning gradient boosted multi-label classification rules.

The algorithm was first published in the following [paper](https://arxiv.org/pdf/2006.13346.pdf):

*Rapp M., Loza Mencía E., Fürnkranz J., Nguyen VL., Hüllermeier E. (2020) Learning Gradient Boosted Multi-label Classification Rules. In: Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2020. Lecture Notes in Computer Science. Springer, Cham*  

If you use the algorithm in a scientific publication, we would appreciate citations to the mentioned paper.

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

In addition, the following features that may speed up training or reduce the memory footprint are currently implemented:
 
* Dense or sparse feature matrices can be used for training and prediction. The use of sparse matrices may speed-up training significantly on some data sets.
* Dense or sparse label matrices can be used for training. The use of sparse matrices may reduce the memory footprint in case of large data sets.
* Multi-threading can be used to parallelize the evaluation of a rule's potential refinements across multiple CPU cores. 

## Project structure

```
|-- data                                Directory that contains several benchmark data sets
    |-- ...
|-- data-synthetic                      Directory that contains several synthetic data sets
    |-- ...
|-- python                              Directory that contains the project's source code
    |-- boomer                          Directory that contains the code for loading data sets and running experiments
        |-- boosting                    Directory that contains the implementation of boosting algorithms
            | ...
        |-- common                      Directory that contains the implementations that all algorithms have in common
            | ...
        |-- seco                        Directory that contains the implementation of separate-and-conquer algorithms 
            | ...
        | ...
    |-- main_boomer.py                  Can be used to start an experiment, i.e., to train and evaluate a model, using BOOMER
    |-- main_generate_synthetic_data.py Can be used to generate synthetic data sets
    |-- main_seco.py                    Can be used to start an experiment, i.e., to train and evaluate a model using the separate-and-conquer algorithm
    |-- setup.py                        Distutil definition of the library for installation via pip
    |-- ...
|-- Makefile                            Makefile for compiling the Cython source files and installing a Python virtual environment
|-- README.md                           This file
|-- settings.zip                        PyCharm settings for syntax highlighting of Cython code
|-- ...
```

## Project setup

The library provided by this project requires Python 3.7 or newer and uses C extensions for Python using [Cython](https://cython.org) to speed up computation and to integrate with parts of the code that are implemented in C++. It is recommended to create a virtual environment using the correct version of Python (which requires that this particular Python version is installed on the host) and providing all dependencies that are required to compile the Cython code (`numpy`, `scipy` and `Cython`). IDEs such as PyCharm may provide an option to create such a virtual environment automatically. For manual installation, the project comes with a Makefile that allows to create a virtual environment via the command
```
make venv
```
This should create a new subdirectory `venv` within the project's root directory.

Unlike pure Python programs, the Cython source files (`.pyx` and `.pxd` files) that are used by the library must be compiled (see [documentation](http://docs.cython.org/en/latest/src/quickstart/build.html) for further details). The compilation can be started using the provided Makefile by running
```
make compile
```
This should result in `.c` files, as well as `.so` files (on Linux) or `.pyd` files (on Windows) be placed in the directory `python/boomer/algorithm/`.

To be able to use the library by any program run inside the virtual environment, it must be installed into the virtual environment together with all of its runtime dependencies (e.g. `scikit-learn`, a full list can be found in `setup.py`). For this purpose, the project's Makefile provides the command 

```
make install
```

*Whenever any Cython source files have been modified, they must be recompiled by running the command "make compile" again and updating the installed package via "make install" afterwards! If compiled Cython files do already exist, only the modified files will be recompiled.*

**Cleanup:** To get rid of any compiled C/C++ files, as well as of the virtual environment, the following command can be used:
```
make clean
``` 
For more fine-grained control, the command `make clean_venv` (for deleting the virtual environment) or `make clean_compile` (for deleting the compiled files) can be used.


**Syntax highlighting in PyCharm:** Unfortunately, the Community Edition of PyCharm does not come with Cython support. To enable proper syntax highlighting of Cython code, the file `settings.zip` in the project's root directory can be imported via `File -> Import Settings`.

## Running experiments

The file `main_boomer.py` allows to run experiments on a specific data set using different configurations of the learning algorithm. The implementation takes care of writing the experimental results into `.csv` files and the learned model can (optionally) be stored on disk to reuse it later. 

In order to run an experiment, the following command line arguments must be provided (most of them are optional):

| Parameter                    | Optional? | Default                    | Description                                                                                                                                                                                                                                                    |
|------------------------------|-----------|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--data-dir`                 | No        | `None`                     | The path of the directory where the data sets are located.                                                                                                                                                                                                     |
| `--dataset`                  | No        | `None`                     | The name of the data set file (without suffix).                                                                                                                                                                                                                |
| `--one-hot-encoding`         | Yes       | `False`                    | `True`, if one-hot-encoding should be used for nominal attributes, `False` otherwise.                                                                                                                                                                          |
| `--output-dir`               | Yes       | `None`                     | The path of the directory into which the experimental results (`.csv` files) should be written.                                                                                                                                                                |
| `--store-predictions`        | Yes       | `False`                    | `True`, if the predictions for the individual test examples should be stored as `.csv` files (they may become very large), `False` otherwise. Does only have an effect if the parameter `--output-dir` is specified.                                           |
| `--print-rules`              | Yes       | `False`                    | `True`, if the induced rules should be printed on the console, `False` otherwise.                                                                                                                                                                              |
| `--store-rules`              | Yes       | `False`                    | `True`, if the induced rules should be stored as a `.txt` file, `False` otherwise. Does only have an effect if the parameter `--output-dir` is specified.                                                                                                      |
| `--print-options`            | Yes       | `None`                     | A dictionary that specifies additional options to be used for printing or storing rules, if the parameter `--print-rules` and/or `--store-rules` is set to `True`, e.g. `{\"print_nominal_values\":True}`.                                                     |
| `--evaluate-training-data`   | Yes       | `False`                    | `True`, if the models should not only be evaluated on the test data, but also on the training data, `False` otherwise.                                                                                                                                         |
| `--model-dir`                | Yes       | `None`                     | The path of the directory where models (`.model` files) are located.                                                                                                                                                                                           |
| `--parameter-dir`            | Yes       | `None`                     | The path of the directory, parameter settings (`.csv` files) should be loaded from.                                                                                                                                                                            |
| `--folds`                    | Yes       | `1`                        | The total number of folds to be used for cross-validation or `1`, if no cross validation should be used.                                                                                                                                                       |
| `--current-fold`             | Yes       | `-1`                       | The cross-validation fold to be performed or `-1`, if all folds should be performed. Must be `-1` or greater than `0`and less or equal to `--folds`. If `--folds` is 1, this parameter is ignored.                                                             |
| `--max-rules`                | Yes       | `1000`                     | The number of rules to be induced or `-1`, if the number of rules should not be restricted.                                                                                                                                                                    |
| `--time-limit`               | Yes       | `-1`                       | The duration in seconds after which the induction of rules should be canceled or `-1`, if no time limit should be used.                                                                                                                                        |
| `--label-sub-sampling`       | Yes       | `None`                     | The name of the strategy to be used for label sub-sampling. Must be `random-label-selection` or `None`. Additional arguments may be provided as a dictionary, e.g. `random-label-selection{\"num_samples\":5}`.                                                |                                                       
| `--instance-sub-sampling`    | Yes       | `bagging`                  | The name of the strategy to be used for instance sub-sampling. Must be `bagging`, `random-instance-selection` or `None`. Additional arguments may be provided as a dictionary, e.g. `bagging{\"sample_size\":0.5}`.                                            |
| `--feature-sub-sampling`     | Yes       | `random-feature-selection` | The name of the strategy to be used for feature sub-sampling. Must be `random-feature-selection` or `None`. Additional arguments may be provided as a dictionary, e.g. `random_feature-selection{\"sample_size\":0.5}`.                                        |
| `--feature-binning`          | Yes       | `None`                     | The name of the strategy to be used for feature binning. Must be `equal-width`, `equal-frequency` or `None`, if no feature binning should be used. Additional arguments may be provided as a dictionary, e.g. `equal-width{\"bin_ratio\":0.5}`.                |
| `--label-binning`            | Yes       | `None`                     | The name of the stretagy to be used for label binning. Must be `equal-width` or `None`, if no label binning should be used. Additional arguments may be provided as a dictionary, e.g. `equal-width{\"bin_ratio\":0.33`                                        |
| `--pruning`                  | Yes       | `None`                     | The name of the strategy to be used for pruning rules. Must be `irep` or `None`.                                                                                                                                                                               |
| `--min-coverage`             | Yes       | `1`                        | The minimum number of training examples that must be covered by a rule. Must be at least `1`.                                                                                                                                                                  |
| `--max-conditions`           | Yes       | `-1`                       | The maximum number of conditions to be included in a rule's body. Must be at least `1` or `-1`, if the number of conditions should not be restricted.                                                                                                          |
| `--max-head-refinements`     | Yes       | `-1`                       | The maximum number of times the head of a rule may be refined. Must be at least `1` or `-1`, if the number of refinements should not be restricted.                                                                                                            |
| `--head-refinement`          | Yes       | `None`                     | The name of the strategy to be used for finding the heads of rules. Must be `single-label`, `full` or `None`. If `None` is used, the most suitable strategy is chosen automatically based on the loss function.                                                |
| `--shrinkage`                | Yes       | `0.3`                      | The shrinkage parameter, a.k.a. the learning rate, to be used. Must be greater than `0` and less or equal to `1`.                                                                                                                                              |
| `--loss`                     | Yes       | `label-wise-logistic-loss` | The name of the loss function to be minimized during training. Must be `label-wise-squared-error-loss`, `label-wise-logistic-loss` or `example-wise-logistic-loss`.                                                                                            |
| `--l2-regularization-weight` | Yes       | `1.0`                      | The weight of the L2 regularization that is applied for calculating the scores that are predicted by rules. Must be at least `0`. If `0` is used, the L2 regularization is turned off entirely, increasing the value causes the model to be more conservative. |
| `--random-state`             | Yes       | `1`                        | The seed to the be used by random number generators.                                                                                                                                                                                                           |
| `--feature-format`           | Yes       | `auto`                     | The format to be used for the feature matrix. Must be `sparse`, if a sparse matrix should be used, `dense`, if a dense matrix should be used, or `auto`, if the format should be chosen automatically.                                                         |
| `--label-format`             | Yes       | `auto`                     | The format to be used for the label matrix. Must be `sparse`, if a sparse matrix should be used, `dense`, if a dense matrix should be used, or `auto`, if the format should be chosen automatically.                                                           |
| `--num-threads`              | Yes       | `1`                        | The number of threads to be used for training. Must be at least `1` or `-1`, if the number of cores available on the machine should be used.                                                                                                                   |
| `--log-level`                | Yes       | `info`                     | The log level to be used. Must be `debug`, `info`, `warn`, `warning`, `error`, `critical`, `fatal` or `notset`.                                                                                                                                                |

In the following, the command for running an experiment using an exemplary configuration can be seen. It uses a virtual environment as discussed in section "Project setup". 

```
venv/bin/python3 python/main_boomer.py --data-dir /path/to/data --output-dir /path/to/results/emotions --model-dir /path/to/models/emotions --dataset emotions --folds 10 --num-rules 1000 --instance-sub-sampling bagging --feature-sub-sampling random-feature-selection --loss label-wise-logistic-loss --shrinkage 0.3 --pruning None --head-refinement single-label
```
