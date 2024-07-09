(runnables)=

# Using Your Own Algorithms

When using the command line API, as described {ref}`here<arguments-basic-usage>`, the Python module of the program to be run must be specified. For running the algorithms provided by this project, the module names `mlrl.boosting` and `mlrl.seco` can be used. However, you can also specify the name of a custom module, or the path to a Python source file, that provides an integration with a machine learning algorithm of your choice.

## Integrating an Algorithm

The module or source file, which is given to the command line API, must contain a class named `Runnable` that extends from {py:class}`mlrl.testbed.BaseRunnable <mlrl.testbed.runnables.LearnerRunnable>`. If you want to use a different class name, you can specify a different one via the command line arguments `-r` or `--runnable` as described {ref}`here<arguments-basic-usage>`. Besides the name of the machine learning algorithm to be integrated, the class must override the method {py:meth}`create_classifier <mlrl.testbed.runnables.LearnerRunnable.create_classifier>`. It must return a scikit-learn compatible [estimator](https://scikit-learn.org/stable/glossary.html#term-estimators) to be used in experiments.

In the following, we provide an exemplary implementation of such a class using scikit-learn's {py:class}`sklearn.ensemble.RandomForestClassifier`:

```python
from mlrl.testbed import BaseRunnable
from sklearn.ensemble import RandomForestClassifier


class Runnable(BaseRunnable):

    def __init__(self):
        super().__init__(learner_name='random-forest-classifier')

    def create_classifier(self, args):
        return RandomForestClassifier()

```

Assuming that the source code shown above is saved to a file named `custom_runnable.py` in the working directory, the command line API can be instructed to use it as follows:

```text
testbed custom_runnable.py --data-dir path/to/datasets/ --dataset dataset-name
```

## Defining Command Line Arguments

To ease the configuration of a machine learning algorithm, for which you created a custom integration, the base class {py:class}`mlrl.testbed.BaseRunnable <mlrl.testbed.runnables.LearnerRunnable>` provides a simple mechanism for defining custom command line arguments by overriding the method {py:meth}`configure_arguments <mlrl.testbed.runnables.Runnable.configure_arguments>`. As illustrated below, the user-specified  values for these arguments can then be retrieved in the method {py:meth}`create_classifier <mlrl.testbed.runnables.LearnerRunnable.create_classifier>`:

```python
from argparse import ArgumentParser

# ...

class Runnable(BaseRunnable):

    # ...

    def configure_arguments(self, parser: ArgumentParser):
        super().configure_arguments(parser)
        parser.add_argument('--n-estimators', type=int, default=100, help='The number of trees in the forest')

    def create_classifier(self, args):
        return RandomForestClassifier(n_estimators=args.n_estimators)

```

The method {py:meth}`configure_arguments <mlrl.testbed.runnables.Runnable.configure_arguments>` receives an {py:class}`argparse.ArgumentParser`, which can be used to define any command line arguments you might need, as described in the [official Python documentation](https://docs.python.org/3/library/argparse.html).

## Providing Version Information

Optionally, you can provide information about the version and authors of your custom program by overriding the method {py:meth}`get_program_info <mlrl.testbed.runnables.Runnable.get_program_info>` of the parent class {py:class}`mlrl.testbed.BaseRunnable <mlrl.testbed.runnables.LearnerRunnable>`:

```python
from typing import Optional
from mlrl.testbed.runnables import LearnerRunnable

# ...

class Runnable(BaseRunnable):

    # ...

    def get_program_info(self) -> Optional[LearnerRunnable.ProgramInfo]:
        return LearnerRunnable.ProgramInfo(name='Random Forest Classifier',
                                           version='1.0.0',
                                           year='1934',
                                           authors=['Bonnie', 'Clyde'])

```
