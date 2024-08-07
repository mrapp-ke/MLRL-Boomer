# Multi-label Separate-and-Conquer Rule Learning Algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/mlrl-seco.svg)](https://badge.fury.io/py/mlrl-seco) [![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)

**Important links:** [Documentation](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/seco/index.html) | [Issue Tracker](https://github.com/mrapp-ke/MLRL-Boomer/issues) | [Changelog](https://mlrl-boomer.readthedocs.io/en/latest/misc/CHANGELOG.html) | [Contributors](https://mlrl-boomer.readthedocs.io/en/latest/misc/CONTRIBUTORS.html) | [Code of Conduct](https://mlrl-boomer.readthedocs.io/en/latest/misc/CODE_OF_CONDUCT.html) | [License](https://mlrl-boomer.readthedocs.io/en/latest/misc/LICENSE.html)

This software package provides an implementation of a **Multi-label Separate-and-Conquer (SeCo) Rule Learning Algorithm** that integrates with the popular [scikit-learn](https://scikit-learn.org) machine learning framework.

The goal of [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) is the automatic assignment of sets of labels to individual data points, for example, the annotation of text documents with topics. The algorithm that is provided by this package uses the SeCo paradigm for learning interpretable rule lists.

## Functionalities

The algorithm that is provided by this project currently supports the following core functionalities to learn a binary classification rules:

- **A large variety of heuristics is available** to assess the quality of candidate rules.
- **Rules may predict for a single label or multiple ones** (which enables to model local label dependencies).
- **Rules can be constructed via a greedy search or a beam search.** The latter may help to improve the quality of individual rules.
- **Sampling techniques and stratification methods** can be used to learn new rules on a subset of the available training examples, features, or labels.
- **Fine-grained control over the specificity/generality of rules** is provided via hyper-parameters.
- **Incremental reduced error pruning** can be used to remove overly specific conditions from rules and prevent overfitting.
- **Sequential post-optimization** may help to improve the predictive performance of a model by reconstructing each rule in the context of the other rules.
- **Native support for numerical, ordinal, and nominal features** eliminates the need for pre-processing techniques such as one-hot encoding.
- **Handling of missing feature values**, i.e., occurrences of NaN in the feature matrix, is implemented by the algorithm.

## Runtime and Memory Optimizations

In addition, the following features that may speed up training or reduce the memory footprint are currently implemented:

- **Sparse feature matrices** can be used for training and prediction. This may speed up training significantly on some data sets.
- **Sparse label matrices** can be used for training. This may reduce the memory footprint in case of large data sets.
- **Sparse prediction matrices** can be used to store predicted labels. This may reduce the memory footprint in case of large data sets.
- **Multi-threading** can be used to parallelize the evaluation of a rule's potential refinements across several features or to obtain predictions for several examples in parallel.

## License

This project is open source software licensed under the terms of the [MIT license](https://mlrl-boomer.readthedocs.io/en/latest/misc/LICENSE.html). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](https://mlrl-boomer.readthedocs.io/en/latest/misc/CONTRIBUTORS.html).
