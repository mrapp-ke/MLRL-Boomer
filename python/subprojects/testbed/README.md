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

This software package provides **mlrl-testbed - a command line utility for running machine learning experiments**. It implements a straightforward, easily configurable, and extensible *workflow* for conducting experiments, including steps such as (but not restricted to) the following:

- loading a dataset
- splitting it into training and test sets
- training one or several models
- evaluating the models' performance
- saving experimental results to output files

The package mlrl-testbed provides the following modes of operation for carrying out different tasks:

- **Single:** In this mode a single experiment is run, involving the steps listed above.
- **Batch:** In batch mode, multiple independent experiments using varying datasets and parameters can be run in an automated manner. By installing the optional package [mlrl-testbed-slurm](https://pypi.org/project/mlrl-testbed-slurm/), the experiments can be run via the [Slurm Workload Manager](https://wikipedia.org/wiki/Slurm_Workload_Manager).
- **Read:** Read mode allows to read the output data produced by a previous experiment and to print it on the console or write it to different output files.
- **Run:** In this mode, previous experiments can be run again, overriding some of their parameters if desired.

# mlrl-testbed

On its own, this package is not very powerful. It is intended as a basis for other packages that build functionality upon it. In fact, it does not make any assumptions about the problem domain or type of machine learning algorithm that should be used in an experiment. Instead, implementations of domain- or algorithm-specific functionality are provided by the extensions discussed below.

## Tabular Machine Learning

The package [mlrl-testbed-sklearn](https://pypi.org/project/mlrl-testbed-sklearn/) adds support for tabular machine learning problems by making use of the [scikit-learn](https://scikit-learn.org) framework. It can easily be installed via the following command (and will pull [mlrl-testbed](https://pypi.org/project/mlrl-testbed/) as a dependency):

```
pip install mlrl-testbed-sklearn
```

### 💡 Example

By writing just a small amount of code, any scikit-learn compatible [estimator](https://scikit-learn.org/stable/glossary.html#term-estimators) can be integrated with mlrl-testbed and used in experiments. For example, the following code integrates scikit-learn's [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier):

```python
from argparse import Namespace
from mlrl.testbed_sklearn.runnables import SkLearnRunnable
from mlrl.util.cli import Argument, IntArgument
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Optional, Set


class Runnable(SkLearnRunnable):

    N_ESTIMATORS = IntArgument(
        '--n-estimators',
        description='The number of trees in the forest',
        default=100,
    )

    def get_algorithmic_arguments(self, known_args: Namespace) -> Set[Argument]:
        return { self.N_ESTIMATORS }

    def create_classifier(self, args: Namespace) -> Optional[ClassifierMixin]:
        return RandomForestClassifier()

    def create_regressor(self, args: Namespace) -> Optional[RegressorMixin]:
        return None  # Not needed in this case

```

The previously integrated algorithm can then be used in experiments controlled via a command line API. Assuming that the source code shown above is saved to a file named `custom_runnable.py` in the working directory, we are now capable of fitting a [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier) to a dataset by using the command below.

```text
mlrl-testbed custom_runnable.py \
    --data-dir path/to/datasets/ \
    --dataset dataset-name \
    --n-estimators 50
```

The above command does not only train a model, but also evaluates it according to common measures and prints the evaluation results. It does also demonstrate how algorithmic parameters can be controlled via command line arguments.

It is also possible to run multiple experiments at once by defining the datasets and algorithmic parameters to be used in the different runs in a YAML file:

```text
mlrl-testbed custom_runnable.py --mode batch --config path/to/config.yaml
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

### 🏁 Advantages

Making use of mlrl-testbed does not only help with the burdens of training and evaluating machine learning models, it can also help making your own methods and algorithms more accessible to users. This is demonstrated by the rule learning algorithms [mlrl-boomer](https://pypi.org/project/mlrl-boomer/) and [mlrl-seco](https://pypi.org/project/mlrl-seco/) that can easily be run via the command line API described above and even extend it with rule-specific functionalities.

### 🔧 Functionalities

The package [mlrl-testbed-sklearn](https://pypi.org/project/mlrl-testbed-sklearn/) provides a command line API that allows configuring and running machine learning algorithms. It allows to apply machine learning algorithms to different datasets and can evaluate their predictive performance in terms of commonly used measures. In detail, it supports the following functionalities:

- Single- and multi-output datasets in the [Mulan](https://github.com/tsoumakas/mulan) and [MEKA](https://waikato.github.io/meka/) format are supported (with the help of the package [mlrl-testbed-arff](https://pypi.org/project/mlrl-testbed-arff/)).
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
