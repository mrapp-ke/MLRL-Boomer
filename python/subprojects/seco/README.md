# Multi-label Separate-and-Conquer Rule Learning Algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This software package provides an implementation of a **Multi-label Separate-and-Conquer (SeCo) Rule Learning Algorithm** that integrates with the popular [scikit-learn](https://scikit-learn.org) machine learning framework.

The goal of [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) is the automatic assignment of sets of labels to individual data points, for example, the annotation of text documents with topics. The algorithm that is provided by this package uses the SeCo paradigm for learning interpretable rule lists.

## Features

The algorithm that is provided by this project currently supports the following core functionalities to learn a binary classification rules:

* A large variety of heuristics is available to assess the quality of candidate rules.
* The rules may predict for a single label or for all labels (which enables to model local label dependencies).
* When learning a new rule, random samples of the training examples, features or labels may be used (including different techniques such as sampling with or without replacement).
* Hyper-parameters that provide fine-grained control over the specificity/generality of rules are available.
* The conditions of rules can be pruned based on a hold-out set.
* The algorithm can natively handle numerical, ordinal and nominal features (without the need for pre-processing techniques such as one-hot encoding).
* The algorithm is able to deal with missing feature values, i.e., occurrences of NaN in the feature matrix.

In addition, the following features that may speed up training or reduce the memory footprint are currently implemented:

* Dense or sparse feature matrices can be used for training and prediction. The use of sparse matrices may speed up training significantly on some data sets.
* Dense or sparse label matrices can be used for training. The use of sparse matrices may reduce the memory footprint in case of large data sets.
* Dense or sparse matrices can be used to store predictions. The use of sparse matrices may reduce the memory footprint in case of large data sets.
* Multi-threading can be used to parallelize the evaluation of a rule's potential refinements across multiple CPU cores.

## License

This project is open source software licensed under the terms of the [MIT license](../../../LICENSE.md). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](../../../CONTRIBUTORS.md).
