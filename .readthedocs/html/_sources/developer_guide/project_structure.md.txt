(project-structure)=

# Project Structure

Currently, this project offers the following software packages:

- **{ref}`Rule learning algorithms <project-structure-rule-learners>`** that follow a unified framework. Due to its modularity, different implementations can be used for the various aspects of the algorithms. This enables to flexibly adjust them to different datasets and use cases.
- A **{ref}`command line utility <project-structure-testbed>`**, referred to as *mlrl-testbed*, which helps to run experiments using different configurations of arbitrary machine learning algorithms and to keep track of output data produced by these experiments.

## Overview

The implementations of the packages are split into several components. A schematic depiction of the project's structure is shown in the following.

```{image} ../_static/developer_guide/project_structure_light.svg
---
align: center
alt: Structure of the project
class: only-light
---
```

```{image} ../_static/developer_guide/project_structure_dark.svg
---
align: center
alt: Structure of the project
class: only-dark
---
```

For maximum efficiency, the core aspects of machine learning algorithms are written in [C++](https://en.wikipedia.org/wiki/C%2B%2B). To provide a user-friendly interface and to integrate the algorithms with the widely used [scikit-learn](https://scikit-learn.org) framework, [Python](<https://en.wikipedia.org/wiki/Python_(programming_language)>) APIs are provided. They rely on the [Cython](https://en.wikipedia.org/wiki/Cython) programming language to interact with the underlying C++ implementation.

(project-structure-rule-learners)=

## Rule Learning Algorithms

By providing a shared library that implements the algorithmic aspects, many rule learning algorithms have in common ("libmlrlcommon"), the implementation of different types of learning methods ("libmlrlboosting", "libmlrlseco", and possibly others in the future) is facilitated. This is especially relevant for developers or scientists who want to build upon the project's source code for developing novel machine learning approaches (see {ref}`cpp-apidoc`).

The low-level C++ libraries are used by higher-level Python packages. Again, this includes a module that provides common functionality ("mlrl-common"), as well as packages that correspond to specific instantiations of different learning algorithms ("mlrl-boomer", "mlrl-seco", and possibly others in the future). All of these packages may rely on common functionality provided by the package "mlrl-util". For further information on the available Python packages, refer to the {ref}`Python API reference <python-apidoc>`.

(project-structure-testbed)=

## MLRL-Testbed

The application of machine learning methods, as well as scientific research in this area, typically requires to evaluate and compare the performance of different kinds of learning approaches, to identify optimal parameter settings for a particular use case, or to analyze the decisions that are made by a previously trained model. To reduce the burden that comes with such tasks, this project offers the Python package "mlrl-testbed" that allows to perform experiments using different kinds of machine learning algorithms, including the algorithms developed by this project, via a command line API. It provides means for parameter tuning and eases to collect experimental results in terms of commonly used evaluation measures or the characteristics of models or datasets. Information on how to use these tools can be found in the section {ref}`testbed`.

The package "mlrl-testbed" is agnostic of the problem domain to be solved and the type of machine learning algorithm to be used in experiments. Domain-specific functionality is implemented by the package "mlrl-testbed-sklearn", which adds support for the [scikit-learn](https://scikit-learn.org) framework to the package "mlrl-testbed". Similarly, the package "mlrl-testbed-arff" adds support for [ARFF](https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/) files. All of these packages may rely on common functionality provided by the package "mlrl-util".
