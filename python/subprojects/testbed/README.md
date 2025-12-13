<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/mrapp-ke/MLRL-Boomer/raw/main/doc/_static/logo_testbed_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/mrapp-ke/MLRL-Boomer/raw/main/doc/_static/logo_testbed_light.svg">
    <img alt="mlrl-testbed" src="https://github.com/mrapp-ke/MLRL-Boomer/raw/main/.assets/logo_testbed_light.svg">
  </picture>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/mlrl-testbed.svg)](https://badge.fury.io/py/mlrl-testbed) [![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)

**🔗 Important links:** [Documentation](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/testbed/index.html) | [Issue Tracker](https://github.com/mrapp-ke/MLRL-Boomer/issues) | [Changelog](https://mlrl-boomer.readthedocs.io/en/latest/misc/CHANGELOG.html) | [License](https://mlrl-boomer.readthedocs.io/en/latest/misc/LICENSE.html)

<!-- documentation-start -->

This software package provides **mlrl-testbed - a command line utility for running machine learning experiments**. It implements a *straightforward*, *easily configurable*, and *extensible* workflow for conducting experiments.

# mlrl-testbed

On its own, this package is not very powerful. It is intended as a basis for other packages that build functionality upon it. In fact, it does not make any assumptions about the problem domain or type of machine learning algorithm that should be used in an experiment. Instead, implementations of domain- or algorithm-specific functionality are provided by the extensions discussed below.

## Tabular Machine Learning

The package [mlrl-testbed-sklearn](https://pypi.org/project/mlrl-testbed-sklearn/) adds support for tabular machine learning problems by making use of the [scikit-learn](https://scikit-learn.org) framework. It can easily be installed via the following command (and will pull [mlrl-testbed](https://pypi.org/project/mlrl-testbed/) as a dependency):

```
pip install mlrl-testbed-sklearn
```

### 💡 Example 1: Running an Experiment

After installing the package the machine learning algorithms offered by the scikit-learn project can be configured and experimented with via a command line API. For example, the following command runs an experiment using scikit-learn's [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier):

```text
mlrl-testbed mlrl.testbed_sklearn \
    --data-dir path/to/datasets/ \
    --dataset dataset-name \
    --estimator RandomForestClassifier
    --n-estimators 50
```

The above command does not only train a model, but also evaluates it according to common measures and prints the evaluation results. It also demonstrates how algorithmic parameters, like `--n-estimator`, can be controlled via command line arguments. A description of all available arguments can be obtained by specifying the `--help` flag.

### 💡 Example 2: Using a Meta-Estimator

By using the argument `--meta-estimator`, it is also possible to use so-called "meta-estimators" in experiments. These are algorithms that decompose the given problem in some way and make use of a "base estimator" to solve the subproblems individually. The following example demonstrates how the meta-estimator [ClassifierChain](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain) can be used with the base estimator [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier):

```text
mlrl-testbed mlrl.testbed_sklearn \
    --data-dir path/to/datasets/ \
    --dataset dataset-name \
    --meta-estimator ClassifierChain \
    --estimator DecisionTreeClassifier \
    --max-depth 5 \
    --meta-verbose true
```

Whereas the argument `--max-depth` belongs to the base learner (and controls the maximum depths of individual trees), the argument `--meta-verbose` belongs to the meta-estimator (and instructs it to produce more verbose log output). All arguments associated with meta-estimators start with the "meta" prefix.

### 💡 Example 3: Running Multiple Experiments

It is also possible to run multiple experiments at once by defining the datasets and algorithmic parameters to be used in the different runs in a YAML file (if the package [mlrl-testbed-slurm](https://pypi.org/project/mlrl-testbed-slurm/) is installed, individual experiments can be submitted to a SLURM cluster by appending the argument `--runner slurm`):

```text
mlrl-testbed mlrl.testbed_sklearn \
    --mode batch \
    --estimator RandomForestClassifier \
    --config path/to/config.yaml \
    --base-dir path/to/results
    --store-evaluation true
```

An exemplary YAML file is shown below. Each combination of the specified parameter values is applied to each dataset defined in the file.

```yaml
datasets:
  - directory: path/to/datasets/
    names:
      - first-dataset
      - second-dataset
parameters:
  - name: --n-estimators
    values:
      - 50
      - 100
```

The experimental results produced before, can afterward be viewed by using read mode. The output of the following command does only include results for individual experiments, but also aggregates them into a condensed overview:

```text
mlrl-testbed mlrl.testbed_sklearn \
    --mode read \
    --input-dir path/to/results \
    --print-evaluation true
```

### 🏁 Advantages

Making use of mlrl-testbed does not only help with the burdens of training and evaluating machine learning models, it can also help making your own methods and algorithms more accessible to users. This is demonstrated by the rule learning algorithms [mlrl-boomer](https://pypi.org/project/mlrl-boomer/) and [mlrl-seco](https://pypi.org/project/mlrl-seco/) that can easily be run via the command line API described above and even extend it with rule-specific functionalities.

### 🔧 Functionalities

The package [mlrl-testbed-sklearn](https://pypi.org/project/mlrl-testbed-sklearn/) provides a command line API that allows configuring and running machine learning algorithms. It allows to apply machine learning algorithms to different datasets and can evaluate their predictive performance in terms of commonly used measures. In detail, it supports the following functionalities:

- Single- and multi-output datasets in the [LIBSVM](https://en.wikipedia.org/wiki/LIBSVM) are supported out-of the box. Datasets in the [Mulan](https://github.com/tsoumakas/mulan) and [MEKA](https://waikato.github.io/meka/) format are supported with the help of the package [mlrl-testbed-arff](https://pypi.org/project/mlrl-testbed-arff/).
- Datasets can automatically be split into training and test data, including the possibility to use cross validation. Alternatively, predefined splits can be provided as separate files.
- One-hot-encoding can be applied to nominal or binary features.
- Binary predictions, scores, or probability estimates can be obtained from machine learning algorithms, if supported. Evaluation measures that are suited for the respective type of predictions are picked automatically.

Furthermore, the command line API provides many options for controlling the experimental results to be gathered during an experiment. Depending on the configuration, the following experimental results can be saved to output files or printed on the console:

- Evaluation scores according to commonly used measures
- Characteristics, i.e., statistical properties, of datasets
- Predictions and their characteristics
- Unique label vectors contained in a classification dataset

If the following are written to output files, they can be loaded and reused in future experiments:

- The machine learning models that have been learned
- Algorithmic parameters used for training

<!-- documentation-end -->

## 📚 Documentation

Our documentation provides an extensive [user guide](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/testbed/index.html), as well as an [API reference](https://mlrl-boomer.readthedocs.io/en/latest/developer_guide/api/python/testbed/mlrl.testbed.html) for developers.

- Examples of how to save [experimental results](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/testbed/outputs.html) to output files.
- Instructions for [using your own algorithms](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/testbed/runnables.html) with the command line API.
- An overview of available [command line arguments](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/testbed/arguments.html) for controlling experiments.

For an overview of changes and new features that have been included in past releases, please refer to the [changelog](https://mlrl-boomer.readthedocs.io/en/latest/misc/CHANGELOG.html).

## 📜 License

This project is open source software licensed under the terms of the [MIT license](https://mlrl-boomer.readthedocs.io/en/latest/misc/LICENSE.html). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](https://mlrl-boomer.readthedocs.io/en/latest/misc/CONTRIBUTORS.html).

All contributions to the project and discussions on the [issue tracker](https://github.com/mrapp-ke/MLRL-Boomer/issues) are expected to follow the [code of conduct](https://mlrl-boomer.readthedocs.io/en/latest/misc/CODE_OF_CONDUCT.html).
