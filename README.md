# README

This project provides a scikit-learn implementation of "BOOMER" - an algorithm for learning gradient boosted multi-label classification rules.

## Project structure

```
|-- data                Directory that contains several benchmark data sets
    |-- ...
|-- python              Directory that contains the project's source code
    |-- boomer          Directory that contains the code for loading data sets and running experiments
        |-- algorithm   Directory that contains the actual implementation of the learning algorithm 
            | ...
        | ...
    |-- setup.py        Distutil definition of the library for installation via pip
    |-- main.py         Implements a main function that can be used to start an experiment, i.e., to train a model
    |-- plots.py        Implements another main function that can be used to create various plots for an existing model
|-- Makefile            Makefile for compiling the Cython source files and installing a Python virtual environment
|-- README.md           This file
|-- settings.zip        PyCharm settings for syntax highlighting of Cython code
```

## Project setup

The library provided by this project requires Python 3.7 and uses C extensions for Python using [Cython](https://cython.org) to speed up computation. It is recommended to create a virtual environment using the correct version of Python (which requires that this particular Python version is installed on the host) and providing all dependencies that are required to compile the Cython code (`numpy`, `scipy` and `Cython`). IDEs such as PyCharm may provide an option to create such a virtual environment automatically. For manual installation, the project comes with a Makefile that allows to create a virtual environment via the command
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

The file `main.py` allows to run experiments on a specific data set using different configurations of the learning algorithm. The implementation takes care of writing the experimental results into `.csv` files and the learned model can (optionally) be stored on disk to reuse it later. 

In order to run an experiment, the following command line arguments must be provided (most of them are optional):

| Parameter                 | Optional? | Default              | Description                                                                                                                                                                                                       |
|---------------------------|-----------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--data-dir`              | No        | `None`               | The path of the directory where the data sets are located.                                                                                                                                                        |
| `--output-dir`            | No        | `None`               | The path of the directory into which the experimental results (`.csv` files) should be written.                                                                                                                   |
| `--model-dir`             | Yes       | `None`               | The path of the directory where models (`.model` files) are located.                                                                                                                                              |
| `--parameter-dir`         | Yes       | `None`               | The path of the directory, parameter settings (`.csv` files) should be loaded from.                                                                                                                               |
| `--dataset`               | No        | `None`               | The name of the data set file (without suffix).                                                                                                                                                                   |
| `--folds`                 | Yes       | `1`                  | The total number of folds to be used for cross-validation or `1`, if no cross validation should be used.                                                                                                          |
| `--current-fold`          | Yes       | `-1`                 | The cross-validation fold to be performed or `-1`, if all folds should be performed. Must be `-1` or greater than `0`and less or equal to `--folds`. If `--folds` is 1, this parameter is ignored.                |
| `--store-predictions`     | Yes       | `False`              | `True`, if the predictions for the individual test examples should be stored as `.csv` files (they may become very large), `False` otherwise.                                                                     |
| `--random-state`          | Yes       | `1`                  | The seed to the be used by random number generators.                                                                                                                                                              |
| `--num-rules`             | Yes       | `100`                | The number of rules to be induced per iteration.                                                                                                                                                                  |
| `--instance-sub-sampling` | Yes       | `None`               | The name of the strategy to be used for instance sub-sampling. Must be `bagging`, `random-instance-selection` or `None`.                                                                                          |
| `--feature-sub-sampling`  | Yes       | `None`               | The name of the strategy to be used for feature sub-sampling. Must be `random-feature-selection` or `None`.                                                                                                       |
| `--pruning`               | Yes       | `None`               | The name of the strategy to be used for pruning rules. Must be `irep` or `None`.                                                                                                                                  |
| `--head-refinement`       | Yes       | `None`               | The name of the strategy to be used for finding the heads of rules. Must be `single-label`, `full` or `None`. If `None` is used, the most suitable strategy is chosen automatically based on the loss function.   |
| `--shrinkage`             | Yes       | `1.0`                | The shrinkage parameter to be used. Must be greater than `0` and less or equal to `1`.                                                                                                                            |
| `--loss`                  | Yes       | `squared-error-loss` | The name of the loss function to be minimized during training. Must be `squared-error-loss` or `logistic-loss`.                                                                                                   |

In the following, the command for running an experiment using an exemplary configuration can be seen. It uses a virtual environment as discussed in section "Project setup". 

```
venv/bin/python3.7 python/main.py --data-dir /path/to/data --output-dir /path/to/results/emotions --model-dir /path/to/models/emotions --dataset emotions --folds 10 --num-rules 1000 --instance-sub-sampling bagging --feature-sub-sampling random-feature-selection --loss squared-error-loss --shrinkage 0.25 --pruning None --head-refinement single-label
```
