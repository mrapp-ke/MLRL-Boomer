(testbed-introduction)=

# Introduction

mlrl-testbed is a command line tool designed to support researchers in conducting empirical studies with a wide range of machine learning algorithms. It provides a *straightforward*, *easily configurable*, and *extensible* workflow for conducting experiments, including steps such as, but not restricted to, the following:

- loading a dataset
- splitting it into training and test sets
- training one or more models
- evaluating the models' predictive performance
- saving experimental results to output files

By default, mlrl-testbed performs a single experiment with a specific dataset and parameter setting. However, for carrying out different tasks, mlrl-testbed can also be operated in the following modes:

- **{ref}`Batch mode <testbed-batch-mode>`:** Allows running multiple independent experiments with varying datasets and parameter settings in an automated manner. Installing the optional package [mlrl-testbed-slurm](https://pypi.org/project/mlrl-testbed-slurm/) enables to run experiments via the [Slurm Workload Manager](https://wikipedia.org/wiki/Slurm_Workload_Manager).
- **{ref}`Read mode <testbed-read-mode>`:** Loads the output of previously executed experiments and prints it to the console or writes it to new output files. This mode is particularly useful for inspecting results obtained via batch mode, as results are automatically aggregated across different experiments.
- **{ref}`Run mode <testbed-run-mode>`:** Allows re-running previously conducted experiments, with the option to override parts of their configuration.

## Command Line Interface

All commands for executing mlrl-testbed follow the following scheme:

```text
mlrl-testbed <runnable> [mode] <control arguments> [control arguments] [hyperparameters]
```

Mandatory arguments (surrounded by `<` and `>`, whereas optional arguments are denoted by `[` and `]`) must always be included by a command. These include arguments for specifying a so-called *runnable*. This term refers to a Python source file or module implementing a simple API in order to integrate one or several algorithms with mlrl-testbed, and possibly extending it with additional functionality. This enables to integrate custom algorithms with little effort, as described {ref}`here <runnables>`. However, for tabular machine learning problems, we provide out-of-the-box support via the package [mlrl-testbed-sklearn](https://pypi.org/project/mlrl-testbed-sklearn/) that relies on the well-established [scikit-learn](https://scikit-learn.org) framework. It can easily be installed via a Python package manager, such as [pip](<https://en.wikipedia.org/wiki/Pip_(package_manager)>) (and will pull [mlrl-testbed](https://pypi.org/project/mlrl-testbed/) as a dependency):

```text
python -m pip install mlrl-testbed-sklearn
```

We further distinguish between {ref}`control arguments <arguments-control>` and {ref}`hyperparameters <setting-algorithmic-parameters>`. Arguments belonging to the former category can be mandatory and are used for controlling the behavior of experiments. The arguments for setting an algorithm's hyperparameters depend on the runnable and are always optional. If none of them is specified, the algorithm's default configuration is used.

## Technical Overview

Depending on the runnable and the mode of operation (some steps may be unnecessary in certain modes), mlrl-testbed follows the experimental procedure outlined in the illustration below.

```{image} ../../_static/workflow_testbed_light.svg
---
align: center
alt: Workflow of an experiment
class: only-light
---
```

```{image} ../../_static/workflow_testbed_dark.svg
---
align: center
alt: Workflow of an experiment
class: only-dark
---
```

Each experiment starts by loading input data from different *sources*. For example, depending on the mode, datasets may be read from {ref}`LIBSVM <dataset-format-svm>` or {ref}`ARFF <dataset-format-arff>` files, hyperparameter settings may be read from [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files, previously trained models may be loaded to avoid re-training, and more. After it has finished and depending on its configuration, an experiment might write output data to so-called *sinks*, e.g., the console log or output files. For example, this may include trained models, the hyperparameters that have been used for training, performance statistics according to common measures, the predictions provided by models, statistics about the dataset, and so forth. The sources and sinks can be configured in a fine-grained manner via control arguments as described {ref}`here <testbed-outputs>`. If any output files are produced by an experiment, a meta-data file is automatically created. It holds all information required by mlrl-testbed to process the files in {ref}`read mode <testbed-read-mode>`.

The workflow followed by an experiment can be viewed as a tree, where each node is associated with a state. The inputs read from different sources make up the initial state at the root node. This state is passed down the tree and may be extended at each node by newly gathered data. For example, before training any machine learning models, the dataset is split into distinct training and test sets following a {ref}`configurable procedure <evaluation-data-splitting>`, such as a {ref}`cross validation <cross-validation>`, to be able to obtain unbiased performance estimates later on. For each fold, the training and test sets to be used are put into a copy of the state and passed down to a corresponding child node, where the training procedure is invoked. After models have been trained on a training set, they are passed to child nodes in the same manner as described before, where they can be used to obtain predictions for one or several test sets. These predictions are included in the final state, associated with a leaf of the workflow tree, from which output data, such as evaluation results, are extracted. Out-of-the-box, we support a wide variety of evaluation measures for assessing the quality of {ref}`different types of predictions <prediction-types>` common in tabular {ref}`classification <user-guide-classification>` and {ref}`regression <user-guide-regression>` problems.
