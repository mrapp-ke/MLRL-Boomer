# "MLRL-Testbed": Utilities for Evaluating Multi-label Rule Learning Algorithms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/mlrl-testbed.svg)](https://badge.fury.io/py/mlrl-testbed) [![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)

**Important links:** [Documentation](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/testbed/index.html) | [Issue Tracker](https://github.com/mrapp-ke/MLRL-Boomer/issues) | [Changelog](https://mlrl-boomer.readthedocs.io/en/latest/misc/CHANGELOG.html) | [Contributors](https://mlrl-boomer.readthedocs.io/en/latest/misc/CONTRIBUTORS.html) | [Code of Conduct](https://mlrl-boomer.readthedocs.io/en/latest/misc/CODE_OF_CONDUCT.html) | [License](https://mlrl-boomer.readthedocs.io/en/latest/misc/LICENSE.html)

This software package provides **utilities for training and evaluating machine learning algorithms**, including classification and regression problems.

## :wrench: Functionalities

Most notably, the package includes a command line API that allows configuring and running machine learning algorithms. For example, the [BOOMER algorithm](https://mlrl-boomer.readthedocs.io/en/stable/user_guide/boosting/index.html) integrates with the command line API out-of-the-box. For [using other algorithms](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/testbed/runnables.html) only a few lines of Python code are necessary.

The command line API allows applying machine learning algorithms to different datasets and evaluating their predictive performance in terms of commonly used measures (provided by the [scikit-learn](https://scikit-learn.org/) framework). In detail, it supports the following functionalities:

- **Single- and multi-output datasets in the [Mulan](http://mulan.sourceforge.net/format.html) and [Meka format](https://waikato.github.io/meka/datasets/)** are supported.
- **Datasets can automatically be split into training and test data, including the possibility to use cross validation.** Alternatively, predefined splits can be used by supplying the data as separate files.
- **One-hot-encoding** can be applied to nominal or binary features.
- **Binary predictions, scores, or probability estimates** can be obtained from a machine learning algorithm, including **classification** and regression algorithms. Evaluation measures that are suited for the respective type of predictions are picked automatically.
- **Evaluation scores can be saved** to output files and printed on the console.
- **Ensemble models can be evaluated incrementally**, i.e., they can be evaluated repeatedly using only a subset of their members with increasing size.
- **Textual representations of models can be saved** to output files and printed on the console. In addition, the **characteristics of models can also be saved** and printed.
- **Characteristics of datasets can be saved** to output files and printed on the console.
- **Unique label vectors contained in a classification dataset can be saved** to output files and printed on the console.
- **Predictions can be saved** to output files and printed on the console. In addition, **characteristics of predictions can also be saved** and printed.
- **Models for the calibration of probabilities can be saved** to output files and printed on the console.
- **Models can be saved on disk** in order to be reused by future experiments.
- **Algorithmic parameters can be read from configuration files** instead of providing them via command line arguments. When providing parameters via the command line, corresponding configuration files can automatically be saved on disk.

## :scroll: License

This project is open source software licensed under the terms of the [MIT license](https://mlrl-boomer.readthedocs.io/en/latest/misc/LICENSE.html). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](https://mlrl-boomer.readthedocs.io/en/latest/misc/CONTRIBUTORS.html).
