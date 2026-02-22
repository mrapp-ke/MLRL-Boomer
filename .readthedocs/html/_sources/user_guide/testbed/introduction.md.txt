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
- **{ref}`Read mode <testbed-read-mode>`:** Loads the output of previously executed experiments and prints it to the console or writes it to new output files. This mode is particularly useful for inspecting results obtained via batch mode as results are automatically aggregated across different experiments.
- **{ref}`Run mode <testbed-run-mode>`:** Allows re-running previously conducted experiments with the option to override parts of their configuration.

## Statement of Need

The growing interest in technology related to artificial intelligence has resulted in a large number of tools for training and evaluating machine learning models and tracking the results of experiments, often without requiring deep technical knowledge or programming skills. Most prominently, this includes commercial and proprietary services like [Google AutoML](https://cloud.google.com/automl), [H2O Driverless AI](https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/), [neptune.ai](https://neptune.ai), or [Comet.ML](https://www.comet.com). These typically provide a web-based interface, nowadays often have a strong focus on large language models (LLMs) rather than tabular machine learning, usually provide a rich feature set with visualization tools and [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning) features, and often offer to purchase computational resources. They might be not free to use or limited in features for non-paying users. Some commercial products are available under open source licenses, such as [MLflow](https://github.com/mlflow/mlflow)[^zaharia2018], [Weights and Biases](https://github.com/wandb), or [KNIME](https://www.knime.com/knime-analytics-platform). Non-commercial open source solutions tend to focus on specific problems of the machine learning toolchain. For example, [WEKA](https://github.com/weka)[^markov2006] and [Orange](https://github.com/biolab/orange3)[^demsar2013] are desktop applications for orchestrating data mining pipelines with algorithms included in the respective software, [TensorBoard](https://www.tensorflow.org/tensorboard) provides a web interface for visualizing experimental results, [PyExperimenter](https://github.com/tornede/py_experimenter)[^tornede2023] helps with distributing jobs and keeping track of results, [Sacred](https://github.com/IDSIA/sacred)[^greff2017] is a Python programming framework for tracking experiments and their results, [DataVersionControl](https://github.com/treeverse/dvc)[^barrak2021] implements a version control system for models and data, and [Sumatra](https://github.com/open-research/sumatra)[^davison2018] is a tool for running command line programs and keeping track of their outputs.

As a lightweight and cross-platform command line utility, mlrl-testbed aims to fill a niche in this open source landscape: Via a straight-forward, but feature-rich, command line interface, it allows to flexibly configure and run experiments in a reproducible manner, resulting in consistently named and organized output files. It can be used either by hand or in scripts and, because it is a Python package, installing and using it is possible on most systems, including headless ones. This makes mlrl-testbed a versatile tool for usage in more complex workflows. For example, although we offer a solution for running distributed experiments via our {ref}`SLURM integration <testbed-slurm>`, other tools may be used for (distributed) job management, starting different experiments via mlrl-testbed as needed. Similarly, one has free choice when it comes to data management tools for handling the output files produced by such runs. Rather than providing any algorithms on its own, mlrl-testbed focuses on the integration of well-tested algorithms offered by other open source projects into a single workflow. Algorithms provided by the [scikit-learn](https://scikit-learn.org) project are one example supported out-of-the-box. Due to the modular design discussed below, and the nature of open source software, integrations of other methods or even support for different machine learning domains can freely be developed and shared publicly by third parties.

## Command Line Interface

All commands for executing mlrl-testbed follow the following scheme:

```text
mlrl-testbed <runnable> [mode] <control arguments> [control arguments] [hyperparameters]
```

Mandatory arguments (surrounded by `<` and `>`, whereas optional arguments are enclosed by `[` and `]`) must always be included by a command. These include arguments for specifying a so-called *runnable*. This term refers to a Python source file or module implementing a simple API in order to integrate one or several algorithms with mlrl-testbed, and possibly extending it with additional functionality. This enables to integrate custom algorithms with little effort, as described {ref}`here <runnables>`. However, for tabular machine learning problems, we provide out-of-the-box support via the package [mlrl-testbed-sklearn](https://pypi.org/project/mlrl-testbed-sklearn/) that relies on the well-established [scikit-learn](https://scikit-learn.org) framework. It allows configuring and running the machine learning algorithms offered by this framework without the need to write any code. The package [mlrl-testbed-sklearn](https://pypi.org/project/mlrl-testbed-sklearn/) can easily be installed via a Python package manager, such as [pip](<https://en.wikipedia.org/wiki/Pip_(package_manager)>) (and will pull [mlrl-testbed](https://pypi.org/project/mlrl-testbed/) as a dependency):

```text
python -m pip install mlrl-testbed-sklearn
```

We further distinguish between {ref}`control arguments <arguments-control>` and {ref}`hyperparameters <setting-algorithmic-parameters>`. Arguments belonging to the former category can be mandatory and are used for controlling the behavior of experiments. The arguments for setting an algorithm's hyperparameters depend on the runnable and are always optional. If none of them is specified, the algorithm's default configuration is used.

## Technical Overview

Depending on the runnable and the mode of operation (some steps may be unnecessary in certain modes), mlrl-testbed follows the experimental procedure outlined in the illustration below.

```{image} ../../_static/user_guide/testbed/workflow_testbed_light.svg
---
align: center
alt: Workflow of an experiment
class: only-light
---
```

```{image} ../../_static/user_guide/testbed/workflow_testbed_dark.svg
---
align: center
alt: Workflow of an experiment
class: only-dark
---
```

Each experiment starts by loading input data from different *sources*. For example, depending on the mode, datasets may be read from {ref}`LIBSVM <dataset-format-svm>` or {ref}`ARFF <dataset-format-arff>` files, hyperparameter settings may be read from [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files, previously trained models may be loaded to avoid re-training, and more. After it has finished and depending on its configuration, an experiment might write output data to so-called *sinks*, e.g., the console log or output files. For example, this may include trained models, the hyperparameters that have been used for training, performance statistics according to common measures, the predictions provided by models, statistics about the dataset, and so forth. The sources and sinks can be configured in a fine-grained manner via control arguments as described {ref}`here <testbed-outputs>`. If any output files are produced by an experiment, a meta-data file is automatically created. It holds all information required by mlrl-testbed to process the files in {ref}`read mode <testbed-read-mode>`.

The workflow followed by an experiment can be viewed as a tree, where each node is associated with a state. The inputs read from different sources make up the initial state at the root node. This state is passed down the tree and may be extended at each node by newly gathered data. For example, before training any machine learning models, the dataset is split into distinct training and test sets following a {ref}`configurable procedure <evaluation-data-splitting>`, such as a {ref}`cross validation <cross-validation>`, to be able to obtain unbiased performance estimates later on. For each fold, the training and test sets to be used are put into a copy of the state and passed down to a corresponding child node, where the training procedure is invoked. After models have been trained on a training set, they are passed to child nodes in the same manner as described before, where they can be used to obtain predictions for one or several test sets. These predictions are included in the final state, associated with a leaf of the workflow tree, from which output data, such as evaluation results, are extracted. Out-of-the-box, we support a wide variety of evaluation measures for assessing the quality of {ref}`different types of predictions <prediction-types>` common in tabular {ref}`classification <user-guide-classification>` and {ref}`regression <user-guide-regression>` problems.

[^zaharia2018]: Matei Zaharia, Andrew Chen, Aaron Davidson, Ali Ghodsi, Sue Ann Hong, Andy Konwinski, Siddharth Murching, Tomas Nykodym, Paul Ogilvie, Mani Parkhe, Fen Xie, and Corey Zumar (2018). ‘Accelerating the machine learning lifecycle with MLflow’. In: *IEEE Data Engineering Bulletin* 41.4, pp. 39–45.

[^markov2006]: Zdravko Markov and Ingrid Russell (2006). ‘An introduction to the WEKA data mining system’. In: *ACM SIGCSE Bulletin* 38.3, pp. 367-368.

[^demsar2013]: Janez Demšar, Tomaž Curk, Aleš Erjavec, Črt Gorup, Tomaž Hočevar, Mitar Milutinovič, Martin Možina, Matija Polajnar, Marko Toplak, Anže Starič, Miha Štajdohar, Lan Umek, Lan Žagar, Jure Žbontar, Marinka Žitnik, and Blaž Zupan (2013). ‘Orange: Data Mining Toolbox in Python’ In: *Journal of Machine Learning Research* 14.1, pp 2349-2353.

[^tornede2023]: Tanja Tornede, Alexander Tornede, Lukas Fehring, Lukas Gehring, Helena Graf, Jonas Hanselle, Felix Mohr, and Marcel Wever (2023). ‘PyExperimenter: Easily distribute experiments and track results’. In: *Journal of Open Source Software*, 8.84, pp. 5149.

[^greff2017]: Klaus Greff, Aaron Klein, Martin Chovanec, Frank Hutter, and Jürgen Schmidhuber, ‘The Sacred Infrastructure for Computational Research’. In: \*Proc. Python in Science Conference, 2017, pp. 49–56.

[^barrak2021]: Amine Barrak, Ellis E. Eghan, and Bram Adams (2021). ‘On the Co-evolution of ML Pipelines and Source Code - Empirical Study of DVC Projects’. In: *Proc. IEEE International Conference on Software Analysis, Evolution, and Reengineering*, pp. 422-433.

[^davison2018]: Andrew P. Davison, Michele Mattioni, Dmitry Smarkanov, and Bartosz Teleńczuk (2018). ‘Sumatra: a toolkit for reproducible research’. *Chapman and Hall/CRC*, 57-78.
