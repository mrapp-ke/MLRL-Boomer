(installation)=

# Installation

All algorithms provided by this project are published on [PyPi](https://pypi.org/). As shown below, they can easily be installed via the Python package manager [pip](<https://en.wikipedia.org/wiki/Pip_(package_manager)>). Unless you intend to modify the algorithms' source code, in which case you should have a look at the section {ref}`compilation`, this is the recommended way for installing the software.

```{note}
Currently, the packages mentioned below are available for Linux (x86_64 and aarch64), macOS (arm64) and Windows (AMD64).
```

Examples of how to use the algorithms in your own Python programs can be found in the section {ref}`usage`.

## Installing the BOOMER Algorithm

The gradient boosting algorithm BOOMER is published as the Python package [mlrl-boomer](https://pypi.org/project/mlrl-boomer/). It can be installed via the following command:

```text
pip install mlrl-boomer
```

The description of the methodology used by the BOOMER algorithm, as well as examples of how to configure it, are given {ref}`here<user-guide-boomer>`.

## Installing the SeCo Algorithm

In addition to the BOOMER algorithm, this project does also provide a Separate-and-Conquer (SeCo) algorithm based on traditional rule learning techniques that are particularly well-suited for learning interpretable models. It is published as the package [mlrl-seco](https://pypi.org/project/mlrl-seco/) and can be installed as follows:

```text
pip install mlrl-seco
```

In {ref}`this<user-guide-seco>` section, we elaborate on the techniques utilized by the SeCo algorithm and discuss its parameters.

## Installing the Command Line API

To ease the use of the algorithms that are developed by this project, we also provide a command line utility that allows configuring the algorithms and applying them to a given dataset without the need to write code. It is published as the Python package [mlrl-testbed](https://pypi.org/project/mlrl-testbed/) and can optionally be installed via the following command:

```text
pip install mlrl-testbed
```

For more information about how to use the command line API, refer to the section {ref}`testbed`.
