---
title: 'mlrl-testbed: A command line utility for tabular machine learning experiments'
tags:
  - Python
  - machine learning
  - scikit-learn
  - experiments
  - evaluation results
authors:
  - name: Michael Rapp
    orcid: 0000-0001-8570-8240
    affiliation: 1
affiliations:
 - name: Independent Researcher, Germany
   index: 1
date: 14 December 2025
bibliography: paper.bib
---

# Summary

The Python package [mlrl-testbed](https://pypi.org/project/mlrl-testbed/) provides a command line utility designed to support researchers in conducting reproducible machine learning experiments. It offers a *straightforward*, *easily configurable*, and *extensible* workflow that supports the full experimental lifecycle:

* Loading a dataset.
* Splitting it into training and test sets.
* Training one or more models.
* Evaluating the models' predictive performance.
* Saving experimental results to output files.

By default, mlrl-testbed executes a single experiment using a given dataset and parameter setting. However, it can also be operated in the following modes:

* **Batch mode:** Allows running multiple independent experiments with varying datasets and parameter settings. Installing the optional package [mlrl-testbed-slurm](https://pypi.org/project/mlrl-testbed-slurm/) enables experiments to be run via the *Slurm Workload Manager*[^1].
* **Read mode:** Allows inspecting the results of previous experiments and saving them to new output files. When viewing results obtained in batch mode, results are automatically aggregated across different experiments.
* **Run mode:** Allows re-running previously conducted experiments with the option to partly override their configuration. Experiments for which results are already available can be skipped.

Originally developed to support work on the BOOMER algorithm [@rapp2020; @rapp2021], mlrl-testbed has since evolved into a standalone utility for empirical machine learning studies.

# State of the Field

The rapid growth of machine learning research has led to a variety of tools for evaluating machine learning methods and tracking the results of empirical experiments. Most prominently, this includes commercial platforms like *Google AutoML*[^2], *H2O Driverless AI*[^3], *neptune.ai*[^4], or *Comet.ML*[^5]. They typically offer a web-based interface with a rich feature set, including visualization tools, AutoML features, and more. While convenient, these tools are proprietary, focus increasingly on large language models rather than tabular machine learning, and may restrict functionality for non-paying users. Some commercial products are available under open source licenses, such as *MLflow* [@zaharia2018], *Weights and Biases*[^6], or *KNIME*[^7]. Open source alternatives tend to focus on specific problems of the machine learning toolchain. For example, desktop applications like *WEKA* [@markov2006] and *Orange* [@demvsar2013] focus on interactive pipeline construction with algorithms included in the respective software. *DataVersionControl* [@barrak2021] implements a version control system for models and data, *TensorBoard*[^8] specializes in visualization, and tools like *PyExperimenter* [@tornede2023], *Sacred* [@greff2017], or *Sumatra* [@davison2018] help with job distribution and keeping track of experimental results.

# Statement of Need

Our software *mlrl-testbed* is an open source tool for researchers, free from any commercial interests, and open to contributions that make it more useful for a broader audience. As a lightweight and cross-platform command line utility, mlrl-testbed complements the existing software ecosystem by addressing a specific niche, rather than extending or replacing any of the tools mentioned above. The straightforward but feature-rich command line interface allows users to flexibly configure and run experiments in a reproducible manner. It can be used interactively or in scripts as part of larger workflows. Because it is distributed as a Python package, it can easily be installed on most systems, including headless servers and high-performance computing environments.

Rather than implementing any machine learning algorithms itself, mlrl-testbed focuses on integrating existing and custom algorithms into a unified workflow. By using a unified methodology for experiments, results obtained with different algorithms can be compared more consistently. It also reduces the burden for researchers, who otherwise must manage experimental trials manually or write scripts that automate this process. Custom algorithms, or even experimental procedures, can easily be integrated by implementing a simple API. Out-of-the-box support is provided for algorithms from the *scikit-learn* [@pedregosa2011] ecosystem. By sharing the mlrl-testbed commands used for experiments, researchers can make their empirical studies more reproducible, as these commands can easily be run on other systems.

# Usage

All commands for executing mlrl-testbed follow the following scheme:

```text
mlrl-testbed <runnable> [mode] <[control arguments]> [hyperparameters]
```

In contrast to optional arguments (enclosed by `[` and `]`), mandatory arguments (surrounded by `<` and `>`) must always be specified. These include arguments for specifying a *runnable*. This is a Python source file or module implementing a simple API to integrate an algorithm with mlrl-testbed and possibly extend it with additional functionality. This abstraction allows users to integrate custom methods with little effort, as described in our documentation[^9]. For tabular machine learning tasks, no custom code is required: The package [mlrl-testbed-sklearn](https://pypi.org/project/mlrl-testbed-sklearn/) provides a ready-to-use integration with the scikit-learn framework. It can easily be installed via a Python package manager such as *pip*:

```text
python -m pip install mlrl-testbed-sklearn
```

We further distinguish between *control arguments* and *hyperparameters*. Arguments belonging to the former category may be mandatory and are used for controlling the behavior of experiments. The arguments for setting an algorithm’s hyperparameters depend on the runnable and are always optional, using the algorithm’s default settings if omitted.

# Software Design

The software design of mlrl-testbed follows the principle of separating experiment orchestration from algorithm implementation. It defines a common interface through which algorithms can be integrated. This separation allows researchers to use the same experimental infrastructure for different algorithms, while keeping our software independent of specific machine learning approaches.

A central design goal of the software is to balance simplicity and flexibility. To favor simplicity, we focus on a lightweight workflow for empirical machine learning studies, outlined in \autoref{fig:workflow}, rather than building a general-purpose experiment-management platform. The workflow can still flexibly be adjusted by implementing custom runnables that come with their own sources or sinks. The choice of a command line interface, instead of a graphical user interface, also benefits flexibility, as it enables automation via scripts and execution on remote systems.

![Illustration of the workflow implemented by mlrl-testbed.\label{fig:workflow}](workflow.svg)

Each experiment starts by loading input data from different *sources*. For example, datasets may be read from *LIBSVM* or *ARFF* files, hyperparameter settings may be read from *CSV* files, or previously trained models may be loaded to avoid re-training. After it has finished, an experiment might write output data to so-called *sinks*, e.g., the console log or output files. This may include trained models, the hyperparameters used for training, performance statistics according to common measures, the predictions provided by models, statistics about the dataset, and more. The abstraction provided by sources and sinks decouples the experimental procedure from specific storage formats and output destinations.

To handle procedures that branch into multiple execution paths, such as cross-validation or hyperparameter searches, the workflow in \autoref{fig:workflow} is modeled as a tree. Each node of the tree is associated with a state. Inputs read from different sources make up the initial state at the root. This state is passed down the tree and may be extended at each node by newly gathered data. For example, if cross-validation is used to split a dataset into distinct training and test sets, the training and test sets for each fold are put into a copy of the current state and passed down to a corresponding child node. Similarly, after models have been trained, they are passed to child nodes, where they can be used to obtain predictions for one or several test sets. These predictions are included in the final state, associated with a leaf of the workflow tree. For assessing the quality of predictions commonly used in tabular classification and regression problems, mlrl-testbed automatically picks a suitable selection of evaluation measures.

# Research Impact Statement

Our software contributes to the reproducibility of empirical machine learning studies by providing a standardized and configurable workflow for conducting experiments. Reproducibility is a widely discussed topic in machine learning research [@pineau2021; @semmelrock2025]. By sharing the commands and configurations used for experiments, researchers can facilitate the reproduction and verification of reported results. Furthermore, mlrl-testbed enables them to focus on developing new methods while relying on a dedicated tool for benchmarking and analysis.

# AI Usage Disclosure

The software project presented in this paper accepts contributions that have been created with the help of AI tools. Contributors are responsible for reviewing all AI-generated content and ensuring its correctness and maintainability. AI-generated contributions are reviewed to the same standards as all other contributions. No AI tools have been used for writing this paper, except for spelling and grammar checking.

[^1]: [https://slurm.schedmd.com/](https://slurm.schedmd.com/)
[^2]: [https://cloud.google.com/automl](https://cloud.google.com/automl)
[^3]: [https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/](https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/)
[^4]: [https://neptune.ai/](https://neptune.ai/)
[^5]: [https://www.comet.com/](https://www.comet.com/)
[^6]: [https://github.com/wandb](https://github.com/wandb)
[^7]: [https://www.knime.com/knime-analytics-platform](https://www.knime.com/knime-analytics-platform)
[^8]: [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)
[^9]: [https://mlrl-boomer.readthedocs.io/en/stable/user_guide/testbed/](https://mlrl-boomer.readthedocs.io/en/stable/user_guide/testbed/)

# References
