<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/mrapp-ke/MLRL-Boomer/raw/main/doc/_static/logo_boomer_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/mrapp-ke/MLRL-Boomer/raw/main/doc/_static/logo_boomer_light.svg">
    <img alt="BOOMER - Gradient Boosted Multi-Label Classification Rules" src="https://github.com/mrapp-ke/MLRL-Boomer/raw/main/.assets/logo_boomer_light.svg">
  </picture>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/mlrl-boomer.svg)](https://badge.fury.io/py/mlrl-boomer) [![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)

**🔗 Important links:** [Documentation](https://mlrl-boomer.readthedocs.io/en/latest/) | [Issue Tracker](https://github.com/mrapp-ke/MLRL-Boomer/issues) | [Changelog](https://mlrl-boomer.readthedocs.io/en/latest/misc/CHANGELOG.html) | [License](https://mlrl-boomer.readthedocs.io/en/latest/misc/LICENSE.html)

<!-- documentation-start -->

This software package provides the official implementation of **BOOMER - an algorithm for learning gradient boosted multi-output rules** that uses [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) for learning an ensemble of rules that is built with respect to a specific multivariate loss function. It integrates with the popular [scikit-learn](https://scikit-learn.org) machine learning framework.

The problem domains addressed by this algorithm include the following:

- **Multi-label classification**: The goal of [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) is the automatic assignment of sets of labels to individual data points, for example, the annotation of text documents with topics.
- **Multi-output regression**: Multivariate [regression](https://en.wikipedia.org/wiki/Regression_analysis) problems require to predict for more than a single numerical output variable.

# The BOOMER Algorithm

To provide a versatile tool for different use cases, great emphasis is put on the *efficiency* of the implementation. Moreover, to ensure its *flexibility*, it is designed in a modular fashion and can therefore easily be adjusted to different requirements. This modular approach enables implementing different kind of rule learning algorithms (see packages [mlrl-common](https://pypi.org/project/mlrl-common/) and [mlrl-seco](https://pypi.org/project/mlrl-seco/)).

## 📖 References

The algorithm was first published in the following [paper](https://doi.org/10.1007/978-3-030-67664-3_8). A preprint version is publicly available [here](https://arxiv.org/pdf/2006.13346.pdf).

*Michael Rapp, Eneldo Loza Mencía, Johannes Fürnkranz Vu-Linh Nguyen and Eyke Hüllermeier. Learning Gradient Boosted Multi-label Classification Rules. In: Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), 2020, Springer.*

If you use the algorithm in a scientific publication, we would appreciate citations to the mentioned paper.

## 🔧 Functionalities

The algorithm that is provided by this project currently supports the following core functionalities for learning ensembles of boosted classification or regression rules.

### Deliberate Loss Optimization

- **Decomposable or non-decomposable loss functions** can be optimized in expectation.
- **L1 and L2 regularization** can be used.
- **Shrinkage (a.k.a. the learning rate) can be adjusted** for controlling the impact of individual rules on the overall ensemble.

### Different Prediction Strategies

- **Various strategies for predicting scores, binary labels or probabilities** are available, depending on whether a classification or regression model is used.
- **Isotonic regression models can be used to calibrate marginal and joint probabilities** predicted by a classification model.

### Flexible Handling of Input Data

- **Native support for numerical, ordinal, and nominal features** eliminates the need for pre-processing techniques such as one-hot encoding.
- **Handling of missing feature values**, i.e., occurrences of *NaN* in the feature matrix, is implemented by the algorithm.

### Fine-grained Control over Model Characteristics

- **Rules can be constructed via a greedy search or a beam search.** The latter may help to improve the quality of individual rules.
- **Single-output, partial, or complete heads** can be used by rules, i.e., they can predict for a single output, a subset of the available outputs, or all of them. Predicting for multiple outputs simultaneously enables to model local dependencies between them.
- **Fine-grained control over the specificity/generality of rules** is provided via hyperparameters.

### Support for Post-Optimization and Pruning

- **Incremental reduced error pruning** can be used for removing overly specific conditions from rules and preventing overfitting.
- **Post- and pre-pruning (a.k.a. early stopping)** allows to determine the optimal number of rules to be included in an ensemble.
- **Sequential post-optimization** may help improving the predictive performance of a model by reconstructing each rule in the context of the other rules.

## ⌚ Runtime and Memory Optimizations

In addition to the features mentioned above, several techniques that may speed up training or reduce the memory footprint are currently implemented.

### Approximation Techniques

- **Unsupervised feature binning** can be used to speed up the evaluation of a rule's potential conditions when dealing with numerical features.
- **Sampling techniques and stratification methods** can be used for learning new rules on a subset of the available training examples, features, or output variables.
- **[Gradient-based label binning (GBLB)](https://arxiv.org/pdf/2106.11690.pdf)** can be used for assigning the labels included in a multi-label classification dataset to a limited number of bins. This may speed up training significantly when minimizing a non-decomposable loss function using rules with partial or complete heads.

### Sparse Data Structures

- **Sparse feature matrices** can be used for training and prediction. This may speed up training significantly on some datasets.
- **Sparse ground truth matrices** can be used for training. This may reduce the memory footprint in case of large datasets.
- **Sparse prediction matrices** can be used for storing predicted labels. This may reduce the memory footprint in case of large datasets.
- **Sparse matrices for storing gradients and Hessians** can be used if supported by the loss function. This may speed up training significantly on datasets with many output variables.

### Parallelization

- **Multi-threading** can be used for parallelizing the evaluation of a rule's potential refinements across several features, updating the gradients and Hessians of individual examples in parallel, or obtaining predictions for several examples in parallel.
- **Single instruction, multiple data (SIMD)** operations supported by modern CPUs are used for parallelizing vector calculations.

### Other Optimizations

- **High- or low-precision data types** can be used for gradients and Hessians to trade off training accuracy and speed differently.

<!-- documentation-end -->

## 📚 Documentation

Our documentation provides an extensive [user guide](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/boosting/index.html), as well as [Python](https://mlrl-boomer.readthedocs.io/en/latest/developer_guide/api/python/boosting/mlrl.boosting.html) and [C++](https://mlrl-boomer.readthedocs.io/en/latest/developer_guide/api/cpp/boosting/filelist.html) API references for developers. If you are new to the project, you probably want to read about the following topics:

- Instructions for [installing the software package](https://mlrl-boomer.readthedocs.io/en/latest/quickstart/installation.html) or [building the project from source](https://mlrl-boomer.readthedocs.io/en/latest/developer_guide/compilation.html).
- Examples of how to [use the algorithm](https://mlrl-boomer.readthedocs.io/en/latest/quickstart/usage.html) in your own Python code or how to use the [command line API](https://mlrl-boomer.readthedocs.io/en/latest/quickstart/testbed.html).
- An overview of available [parameters](https://mlrl-boomer.readthedocs.io/en/latest/user_guide/boosting/parameters.html).

A collection of benchmark datasets that are compatible with the algorithm are provided in a separate [repository](https://github.com/mrapp-ke/Boomer-Datasets).

For an overview of changes and new features that have been included in past releases, please refer to the [changelog](https://mlrl-boomer.readthedocs.io/en/latest/misc/CHANGELOG.html).

## 📜 License

This project is open source software licensed under the terms of the [MIT license](https://mlrl-boomer.readthedocs.io/en/latest/misc/LICENSE.html). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](https://mlrl-boomer.readthedocs.io/en/latest/misc/CONTRIBUTORS.html).

All contributions to the project and discussions on the [issue tracker](https://github.com/mrapp-ke/MLRL-Boomer/issues) are expected to follow the [code of conduct](https://mlrl-boomer.readthedocs.io/en/latest/misc/CODE_OF_CONDUCT.html).
