# README

This project implements "BOOMER" - an algorithm for learning gradient boosted multi-label classification rules.

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

This project requires Python 3.7 and uses C extensions for Python using [Cython](https://cython.org) to speed up computation. Prior to running the Python application, the Cython source files (`.pyx` and `.pxd` files) must be compiled (see [here](http://docs.cython.org/en/latest/src/quickstart/build.html) for further details). The project comes with a Makefile that eases the compilation by running
```
make compile
```
This should result in `.c` files, as well as `.so` files (on Linux) or `.pyd` files (on Windows) be placed in the directory `python/boomer/algorithm/`.

**Syntax highlighting in PyCharm:** Unfortunately, the Community Edition of PyCharm does not come with Cython support. To enable proper syntax highlighting of Cython code, the file `settings.jar` in the project's root directory can be imported via `File -> Import Settings`.

## Running experiments

The file `main.py` allows to run experiments on a specific data set using different configurations of the learning algorithm. The implementation takes case of writing the experimental results into `.csv` files and the learned model can (optionally) be stored on disk to reuse it later. 

In order to run an experiment, the following command line arguments must be provided (most of them are optional):

| Parameter                 | Optional? | Default              | Description                                                                                                                                                                                                    |
|---------------------------|-----------|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--data-dir`              | No        | `None`               | The path of the directory where the data sets are located.                                                                                                                                                     |
| `--output-dir`            | No        | `None`               | The path of the directory into which the experimental results (.csv files) should be written.                                                                                                                  |
| `--model-dir`             | Yes       | `None`               | The path of the directory where models (.model files) are located.                                                                                                                                             |
| `--dataset`               | No        | `None`               | The name of the data set file (without suffix).                                                                                                                                                                |
| `--folds`                 | Yes       | `1`                  | The number of folds to be used for cross-validation or 1, if no cross validation should be used.                                                                                                               |
| `--store-predictions`     | Yes       | `False`              | True, if the predictions for the individual test examples should be stored as .csv files (they may become very large), False otherwise.                                                                        |
| `--random-state`          | Yes       | `1`                  | The seed to the be used by random number generators.                                                                                                                                                           |
| `--num-rules`             | Yes       | `100`                | The number of rules to be induced per iteration.                                                                                                                                                               |
| `--iterations`            | Yes       | `1`                  | The number of iterations. At each iteration new rules are induced and added to the model of the previous iteration.                                                                                            |
| `--instance-sub-sampling` | Yes       | `None`               | The name of the strategy to be used for instance sub-sampling. Must be `bagging`, `random-instance-selection` or `None`.                                                                                       |
| `--feature-sub-sampling`  | Yes       | `None`               | The name of the strategy to be used for feature sub-sampling. Must be `random-feature-selection` or `None`.                                                                                                    |
| `--pruning`               | Yes       | `None`               | The name of the strategy to be used for pruning rules. Must be `irep` or `None`.                                                                                                                               |
| `--head-refinement`       | Yes       | `None`               | The name of the strategy to be used for finding the heads of rules. Must be `single-label`, `full` or `None`. If `None` is used, the most suitable strategy is chose automatically based on the loss function. |
| `--shrinkage`             | Yes       | `1.0`                | The shrinkage parameter to be used. Must be greater than `0` and less or equal to `1`.                                                                                                                         |
