(runnables)=

# Using Your Own Algorithms

When using the package mlrl-testbed, as described {ref}`here<arguments-basic-usage>`, the Python module of the program to be run must be specified. For running the algorithms provided by this project, the module names `mlrl.boosting` and `mlrl.seco` can be used. However, you can also specify the name of a custom module, or the path to a Python source file, that provides an integration with a machine learning algorithm of your choice.

## Integrating an Algorithm

The module or source file, which is specified via the command line API, must contain a class named `Runnable` that extends from {py:class}`mlrl.testbed.SkLearnRunnable <mlrl.testbed.runnables_sklearn.SkLearnRunnable>`. If you want to use a different class name, you can specify a different one via the command line arguments `-r` or `--runnable` as described {ref}`here<arguments-basic-usage>`. Besides the name of the machine learning algorithm to be integrated, the class must override the methods {py:meth}`create_classifier <mlrl.testbed.runnables_sklearn.SkLearnRunnable.create_classifier>` and {py:meth}`create_regressor <mlrl.testbed.runnables_sklearn.SkLearnRunnable.create_regressor>`. If you do not intend to support either classification or regression problems, you can just return `None` from the respective method. Otherwise, it must return a scikit-learn compatible [estimator](https://scikit-learn.org/stable/glossary.html#term-estimators) to be used in experiments.

In the following, we provide an exemplary implementation of such a class using scikit-learn's {py:class}`RandomForestClassifier <sklearn.ensemble.RandomForestClassifier>`:

```python
from argparse import Namespace
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed_sklearn.runnables import SkLearnRunnable
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Optional


class Runnable(SkLearnRunnable):

    def create_classifier(self, mode: ExperimentMode, args: Namespace) -> Optional[ClassifierMixin]:
        return RandomForestClassifier()

    def create_regressor(self, mode: ExperimentMode, args: Namespace) -> Optional[RegressorMixin]:
        return None

```

Assuming that the source code shown above is saved to a file named `custom_runnable.py` in the working directory, the package mlrl-testbed can be instructed to use it as follows:

```text
mlrl-testbed custom_runnable.py --data-dir path/to/datasets/ --dataset dataset-name
```

## Defining Command Line Arguments

To ease the configuration of a machine learning algorithm, for which you created a custom integration, the base class {py:class}`SkLearnRunnable <mlrl.testbed.runnables_sklearn.SkLearnRunnable>` provides a simple mechanism for defining custom command line arguments by overriding the method {py:meth}`get_algorithmic_arguments <mlrl.testbed.runnables.Runnable.get_algorithmic_arguments>`. As illustrated below, the user-specified values for these arguments can then be retrieved in the methods {py:meth}`create_classifier <mlrl.testbed.runnables_sklearn.SkLearnRunnable.create_classifier>` and {py:meth}`create_regressor <mlrl.testbed.runnables_sklearn.SkLearnRunnable.create_regressor>`:

```python
from argparse import Namespace
from mlrl.testbed import SkLearnRunnable
from mlrl.testbed.experiments.state import ExperimentMode
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

    def create_classifier(self, mode: ExperimentMode, args: Namespace) -> Optional[ClassifierMixin]:
        return RandomForestClassifier(n_estimators=self.N_ESTIMATORS.get_value(args))

    def create_regressor(self, mode: ExperimentMode, args: Namespace) -> Optional[RegressorMixin]:
        return None

```

The method {py:meth}`get_algorithmic_arguments <mlrl.testbed.runnables.Runnable.get_algorithmic_arguments>` must return a set of {py:class}`Argument <mlrl.util.cli.Argument>` objects. The following subclasses, corresponding to different types of arguments, are available:

- {py:class}`IntArgument <mlrl.util.cli.IntArgument>`: For specifying an integer value.
- {py:class}`FloatArgument <mlrl.util.cli.FloatArgument>`: For specifying a float value.
- {py:class}`StringArgument <mlrl.util.cli.StringArgument>`: For specifying an arbitrary string.
- {py:class}`SetArgument <mlrl.util.cli.SetArgument>`: For specifying one out of a predefined set of string values.
- {py:class}`EnumArgument <mlrl.util.cli.EnumArgument>`: For specifying one out of a predefined set of enum values.

Instead of retrieving the value specified by the user directly from the given `Namespace` object, we recommend to use the method {py:meth}`get_value <mlrl.util.cli.Argument.get_value>`, as it validates the given value and prints helpful information in the case of validation errors.

## Providing Version Information

Optionally, you can provide information about the version and authors of your custom program by overriding the method {py:meth}`get_program_info <mlrl.testbed.runnables.Runnable.get_program_info>`:

```python
from mlrl.testbed_sklearn.runnables import SkLearnRunnable
from mlrl.testbed.program_info import ProgramInfo
from typing import Optional

class Runnable(SkLearnRunnable):

    # ...

    def get_program_info(self) -> Optional[ProgramInfo]:
        return ProgramInfo(name='Random Forest Classifier',
                           version='1.0.0',
                           year='1934',
                           authors=['Bonnie', 'Clyde'])

```
