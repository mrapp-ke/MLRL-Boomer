"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for running experiments using the scikit-learn framework.
"""
import contextlib
import os
import re as regex

from abc import ABC, abstractmethod
from argparse import Namespace
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Type, override

import docstring_parser
import numpy as np

from sklearn.base import ClassifierMixin as SkLearnClassifierMixin, RegressorMixin as SkLearnRegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.utils import all_estimators

from mlrl.testbed_sklearn.experiments import SkLearnExperiment
from mlrl.testbed_sklearn.experiments.input.dataset.splitters.extension import DatasetSplitterExtension
from mlrl.testbed_sklearn.experiments.output.characteristics.data.extension import TabularDataCharacteristicExtension
from mlrl.testbed_sklearn.experiments.output.characteristics.data.extension_prediction import \
    PredictionCharacteristicsExtension
from mlrl.testbed_sklearn.experiments.output.dataset.extension_ground_truth import GroundTruthExtension
from mlrl.testbed_sklearn.experiments.output.dataset.extension_prediction import PredictionExtension
from mlrl.testbed_sklearn.experiments.output.evaluation.extension import EvaluationExtension
from mlrl.testbed_sklearn.experiments.output.label_vectors.extension import LabelVectorExtension
from mlrl.testbed_sklearn.experiments.prediction import GlobalPredictor
from mlrl.testbed_sklearn.experiments.prediction.extension import PredictionTypeExtension
from mlrl.testbed_sklearn.experiments.prediction.predictor import Predictor
from mlrl.testbed_sklearn.experiments.problem_domain import SkLearnClassificationProblem, SkLearnProblem, \
    SkLearnRegressionProblem

from mlrl.testbed.command import ArgumentList, Command
from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.input.dataset.extension import DatasetFileExtension
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter
from mlrl.testbed.experiments.input.model.extension import ModelInputExtension
from mlrl.testbed.experiments.input.parameters.extension import ParameterInputExtension
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.output.model.extension import ModelOutputDirectoryExtension, ModelOutputExtension
from mlrl.testbed.experiments.output.parameters.extension import ParameterOutputDirectoryExtension, \
    ParameterOutputExtension
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain, RegressionProblem
from mlrl.testbed.experiments.state import ExperimentMode, ExperimentState
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode
from mlrl.testbed.runnables import Runnable
from mlrl.testbed.util.io import ENCODING_UTF8

from mlrl.util.cli import Argument, BoolArgument, FloatArgument, IntArgument, SetArgument
from mlrl.util.format import format_list, format_set


class SkLearnRunnable(Runnable, ABC):
    """
    An abstract base class for all programs that run an experiment using the scikit-learn framework.
    """

    class BatchConfigFile(BatchMode.ConfigFile):
        """
        A YAML configuration file that configures a batch of experiments using the scikit-learn framework to be run.
        """

        def __init__(self, file_path: str):
            """
            :param file_path: The path to the configuration file
            """
            super().__init__(file_path, schema_file_path=Path(__file__).parent / 'batch_config.schema.yml')

        @property
        def dataset_args(self) -> List[ArgumentList]:
            """
            See :func:`from mlrl.testbed.modes.BatchMode.ConfigFile.dataset_args`
            """
            return DatasetFileExtension.parse_dataset_args_from_config(self)

    class GlobalPredictorFactory(SkLearnProblem.PredictorFactory):
        """
        Allow to create instances of type `Predictor` that obtain predictions from a global model.
        """

        def __init__(self, prediction_type: PredictionType):
            """
            :param prediction_type: The type of the predictions to be obtained
            """
            self.prediction_type = prediction_type

        @override
        @override
        def create(self) -> Predictor:
            """
            See :func:`from mlrl.testbed_sklearn.experiments.problem_domain.SkLearnProblem.PredictorFactory.create`
            """
            return GlobalPredictor(self.prediction_type)

    class ProblemDomainExtension(Extension):
        """
        An extension that configures the problem domain.
        """

        PROBLEM_TYPE = SetArgument(
            '--problem-type',
            values={ClassificationProblem.NAME, RegressionProblem.NAME},
            default=ClassificationProblem.NAME,
            description='The type of the machine learning problem to be solved.',
        )

        def __init__(self):
            super().__init__(PredictionTypeExtension())

        @override
        def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
            """
            return {self.PROBLEM_TYPE}

        @override
        def get_supported_modes(self) -> Set[ExperimentMode]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
            """
            return {ExperimentMode.SINGLE, ExperimentMode.BATCH}

        @staticmethod
        def get_problem_domain(args: Namespace,
                               runnable: 'SkLearnRunnable',
                               fit_kwargs: Optional[Dict[str, Any]] = None,
                               predict_kwargs: Optional[Dict[str, Any]] = None) -> ProblemDomain:
            """
            Returns the problem domain that should be tackled by an experiment.

            :param args:            The command line arguments specified by the user
            :param runnable:        The `SkLearnRunnable` that is used to run the experiment
            :param fit_kwargs:      Optional keyword arguments to be passed to the estimator's `predict` function
            :param predict_kwargs:  Optional keyword arguments to be passed to the estimator's `fit` function
            :return:                The problem domain that should be tackled by the experiment
            """
            prediction_type = PredictionTypeExtension.get_prediction_type(args)
            predictor_factory = runnable.create_predictor_factory(args, prediction_type)
            problem_type = SkLearnRunnable.ProblemDomainExtension.PROBLEM_TYPE.get_value(args)

            if problem_type == ClassificationProblem.NAME:
                base_learner = runnable.create_classifier(args)

                if base_learner is None:
                    raise AttributeError('Classification problems are not supported by the runnable "'
                                         + type(runnable).__name__ + '"')

                # pylint: disable=protected-access
                base_learner._validate_params()  # type: ignore[union-attr]
                return SkLearnClassificationProblem(base_learner=base_learner,
                                                    predictor_factory=predictor_factory,
                                                    prediction_type=prediction_type,
                                                    fit_kwargs=fit_kwargs,
                                                    predict_kwargs=predict_kwargs)

            base_learner = runnable.create_regressor(args)

            if base_learner is None:
                raise AttributeError('Regression problems are not supported by the runnable "' + type(runnable).__name__
                                     + '"')

            # pylint: disable=protected-access
            base_learner._validate_params()  # type: ignore[union-attr]
            return SkLearnRegressionProblem(base_learner=base_learner,
                                            predictor_factory=predictor_factory,
                                            prediction_type=prediction_type,
                                            fit_kwargs=fit_kwargs,
                                            predict_kwargs=predict_kwargs)

    @override
    def get_extensions(self) -> List[Extension]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_extensions`
        """
        return [
            SkLearnRunnable.ProblemDomainExtension(),
            DatasetSplitterExtension(),
            PredictionTypeExtension(),
            ModelInputExtension(),
            ModelOutputExtension(),
            ModelOutputDirectoryExtension(),
            ParameterInputExtension(),
            ParameterOutputExtension(),
            ParameterOutputDirectoryExtension(),
            EvaluationExtension(),
            TabularDataCharacteristicExtension(),
            LabelVectorExtension(),
            PredictionExtension(),
            GroundTruthExtension(),
            PredictionCharacteristicsExtension(),
        ] + super().get_extensions()

    @override
    def create_problem_domain(self, args: Namespace) -> ProblemDomain:
        """
        See :func:`mlrl.testbed.experiments.recipe.Recipe.create_problem_domain`
        """
        return SkLearnRunnable.ProblemDomainExtension.get_problem_domain(args, runnable=self)

    @override
    def create_dataset_splitter(self, args: Namespace, load_dataset: bool = True) -> DatasetSplitter:
        """
        See :func:`mlrl.testbed.experiments.recipe.Recipe.create_dataset_splitter`
        """
        return DatasetSplitterExtension.get_dataset_splitter(args, load_dataset)

    @override
    def create_experiment_builder(self,
                                  experiment_mode: ExperimentMode,
                                  args: Namespace,
                                  command: Command,
                                  load_dataset: bool = True) -> Experiment.Builder:
        """
        See :func:`mlrl.testbed.experiments.recipe.Recipe.create_experiment_builder`
        """
        meta_data = MetaData(command=command)
        initial_state = ExperimentState(mode=experiment_mode,
                                        args=args,
                                        meta_data=meta_data,
                                        problem_domain=self.create_problem_domain(args))
        return SkLearnExperiment.Builder(initial_state=initial_state,
                                         dataset_splitter=self.create_dataset_splitter(args, load_dataset))

    @override
    def create_batch_config_file_factory(self) -> BatchMode.ConfigFile.Factory:
        """
        See :func:`mlrl.testbed.runnables.Runnable.create_batch_config_file_factory`
        """
        # pylint: disable=unnecessary-lambda
        return lambda config_file_path: SkLearnRunnable.BatchConfigFile(config_file_path)

    # pylint: disable=unused-argument
    def create_predictor_factory(self, args: Namespace,
                                 prediction_type: PredictionType) -> SkLearnProblem.PredictorFactory:
        """
        May be overridden by subclasses in order to create the `SkLearnProblem.PredictorFactory` that should be used for
        obtaining predictions from a previously trained model.

        :param args:            The command line arguments
        :param prediction_type: The type of the predictions to be obtained
        :return:                The `SkLearnProblem.PredictorFactory` that has been created
        """
        return SkLearnRunnable.GlobalPredictorFactory(prediction_type)

    @abstractmethod
    def create_classifier(self, args: Namespace) -> Optional[SkLearnClassifierMixin]:
        """
        Must be implemented by subclasses in order to create a machine learning algorithm that can be applied to
        classification problems.

        :param args:    The command line arguments
        :return:        The learner that has been created or None, if regression problems are not supported
        """

    @abstractmethod
    def create_regressor(self, args: Namespace) -> Optional[SkLearnRegressorMixin]:
        """
        Must be implemented by subclasses in order to create a machine learning algorithm that can be applied to
        regression problems.

        :param args:    The command line arguments
        :return:        The learner that has been created or None, if regression problems are not supported
        """


class SklearnEstimator:
    """
    Represents a scikit-learn estimator that can be used with a `SklearnEstimatorRunnable`.
    """

    class SklearnArgument(Argument):
        """
        A command line argument that allows to configure a hyperparameter of a scikit-learn estimator.
        """

        def __init__(self,
                     name: str,
                     parameter_name: str,
                     arguments: List[Argument],
                     description: Optional[str] = None,
                     type_hint: Optional[str] = None):
            """
            :param name:            The name of the argument
            :param parameter_name:  The name of the hyperparameter
            :param arguments:       A list that contains arguments that should be used for parsing the argument's value
            :param description:     An optional description of the argument
            :param type_hint:       An optional type hint
            """
            super().__init__(name, description=description)
            self.parameter_name = parameter_name
            self.arguments = arguments
            self.type_hint = type_hint

        @staticmethod
        def from_type_name(argument_name: str,
                           parameter_name: str,
                           type_name: str,
                           description: Optional[str] = None) -> 'SklearnEstimator.SklearnArgument':
            """
            Creates and returns an `SklearnArgument` from a given type name.

            :param argument_name:   The name to be used by the argument
            :param parameter_name:  the name of the hyperparameter
            :param type_name:       The type name to be parsed
            :param description:     An optional description of the argument
            """
            arguments: List[Argument] = []
            type_hints: List[str] = []
            default_value: Optional[str] = None

            for part in [part.strip() for part in regex.split(r'[{}]', type_name) if part]:
                parts2 = [part2.strip() for part2 in regex.split(r',|or', part) if part2]

                if all(part2 == 'None' or (part2.startswith('"') and part2.endswith('"')) for part2 in parts2):
                    values = set(filter(lambda part2: part2 != 'None', map(lambda part2: part2.strip('"'), parts2)))
                    arguments.append(SetArgument(argument_name, values=values))
                    type_hints.append('one of ' + format_set(values))
                else:
                    for part2 in parts2:
                        if part2.startswith('non-negative'):
                            part2 = part2[len('non-negative'):].strip()

                        if part2 == 'int':
                            arguments.insert(0, IntArgument(argument_name))  # Must have priority over 'float'
                            type_hints.append(part2)
                        elif part2 == 'float':
                            arguments.append(FloatArgument(argument_name))
                            type_hints.append(part2)
                        elif part2 == 'bool':
                            arguments.append(BoolArgument(argument_name))
                            type_hints.append(part2)
                        elif part2.startswith('default='):
                            default_value = part2.lstrip('default=')
                        else:
                            raise ValueError('Failed to parse type name: ' + part2)

            type_hint: Optional[str] = None

            if description and type_hints:
                description = description.rstrip() + ('' if description.endswith('.') else '.')
                type_hint = format_list(type_hints, last_separator=' or ')
                description += ' Must be ' + type_hint + '.'

                if default_value:
                    description += ' The default value is ' + default_value + '.'

            return SklearnEstimator.SklearnArgument(name=argument_name,
                                                    parameter_name=parameter_name,
                                                    description=description,
                                                    type_hint=type_hint,
                                                    arguments=arguments)

        @override
        def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
            """
            See :func:`mlrl.util.cli.Argument.get_value`
            """
            for argument in self.arguments:
                try:
                    value = argument.get_value(args, default=None)

                    if value is not None:
                        return value
                except ValueError:
                    pass

            value = super().get_value(args, default=default)

            if value is not None:
                message = 'Invalid value given for argument ' + self.name + '.'
                type_hint = self.type_hint
                message += ' ' + ('Must be' + type_hint + ', but got' if type_hint else 'Got') + ': ' + str(value)
                raise ValueError(message)

            return default

    EstimatorType = Type[SkLearnClassifierMixin] | Type[SkLearnRegressorMixin]

    @staticmethod
    def __format_argument_description(description: str) -> str:
        indent_delimiter = '.. '
        indent = 0
        lines: List[str] = []

        for line in description.split('\n'):
            if line:
                if line.startswith(indent_delimiter):
                    indent = len(indent_delimiter)
                elif indent == 0 or not line.startswith(' ' * indent):
                    lines.append(line.strip().lstrip('- '))
                    indent = 0

        description = ' '.join(lines)
        sentences: List[str] = []

        for sentence in description.split('. '):
            if not regex.search(r':[a-z]+:`.*`', sentence):
                sentences.append(sentence.strip().replace('``', '\'').replace('`', '\''))

        return '. '.join(sentences)

    def __can_be_instantiated(self, *args, **kwargs) -> bool:

        @contextlib.contextmanager
        def suppress_output():
            with open(os.devnull, mode='w', encoding=ENCODING_UTF8) as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    yield

        try:
            with suppress_output():
                instance = self.instantiate(*args, **kwargs)
                rng = np.random.default_rng(seed=1)
                tags = instance.__sklearn_tags__() if hasattr(instance, '__sklearn_tags__') else None
                num_examples = 12
                num_features = 3
                x_shape = (num_examples, num_features)
                x: np.ndarray

                if tags and tags.input_tags.categorical:
                    x = rng.integers(low=0, high=1, endpoint=True, size=x_shape, dtype=np.int32)
                else:
                    x = rng.random(size=x_shape, dtype=np.float32)

                if self.is_regressor:
                    y = rng.random(size=(num_examples, 2 if tags and tags.target_tags.multi_output else 1),
                                   dtype=np.float32)
                else:
                    y = np.zeros(shape=num_examples, dtype=np.uint8)
                    y[(num_examples // 2):] = 1

                    if tags and tags.target_tags.multi_output:
                        y = np.column_stack((y, y))

                instance.fit(x, y)

            return True
        # pylint: disable=broad-exception-caught
        except Exception:
            return False

    def __init__(self, estimator_name: str, estimator_type: EstimatorType):
        """
        :param estimator_name:  The name of the estimator
        :param estimator_type:  The type of the estimator
        """
        self.estimator_name = estimator_name
        self.estimator_type = estimator_type

    @staticmethod
    def get_supported_regressors() -> Set['SklearnEstimator']:
        """
        Returns a set that contains all supported scikit-learn regressors.

        :return: A set that contains the names of all supported regressors
        """
        return set(
            filter(
                lambda estimator: estimator.can_be_default_instantiated, {
                    SklearnEstimator(estimator_name=estimator_name, estimator_type=estimator_type)
                    for estimator_name, estimator_type in all_estimators(type_filter='regressor')
                    if issubclass(estimator_type, SkLearnRegressorMixin)
                }))

    @staticmethod
    def get_supported_classifiers() -> Set['SklearnEstimator']:
        """
        Returns a set that contains all supported scikit-learn classifiers.

        :return: A set that contains the names of all supported classifiers
        """
        return set(
            filter(
                lambda estimator: estimator.can_be_default_instantiated, {
                    SklearnEstimator(estimator_name=estimator_name, estimator_type=estimator_type)
                    for estimator_name, estimator_type in all_estimators(type_filter='classifier')
                    if issubclass(estimator_type, SkLearnClassifierMixin)
                }))

    @staticmethod
    def get_supported_meta_regressors() -> Set['SklearnEstimator']:
        """
        Returns a set that contains all supported scikit-learn meta-regressors.

        :return: A set that contains the names of all supported meta-regressors
        """
        return set(
            filter(
                lambda estimator: estimator.can_be_instantiated_with_estimator, {
                    SklearnEstimator(estimator_name=estimator_name, estimator_type=estimator_type)
                    for estimator_name, estimator_type in all_estimators(type_filter='regressor')
                    if issubclass(estimator_type, SkLearnRegressorMixin)
                }))

    @staticmethod
    def get_supported_meta_classifiers() -> Set['SklearnEstimator']:
        """
        Returns a set that contains all supported scikit-learn meta-classifiers.

        :return: A set that contains the names of all supported meta-classifiers
        """
        return set(
            filter(
                lambda estimator: estimator.can_be_instantiated_with_estimator, {
                    SklearnEstimator(estimator_name=estimator_name, estimator_type=estimator_type)
                    for estimator_name, estimator_type in all_estimators(type_filter='classifier')
                    if issubclass(estimator_type, SkLearnClassifierMixin)
                }))

    @property
    def is_classifier(self) -> bool:
        """
        True, if the estimator is a classifier, False otherwise.
        """
        return issubclass(self.estimator_type, SkLearnClassifierMixin)

    @property
    def is_regressor(self) -> bool:
        """
        True, if the estimator is a regressor, False otherwise.
        """
        return issubclass(self.estimator_type, SkLearnRegressorMixin)

    @property
    def is_meta_estimator(self) -> bool:
        """
        True, if the estimator is a meta-estimator, False otherwise.
        """
        return self.can_be_instantiated_with_estimator

    @cached_property
    def algorithmic_arguments(self) -> Set['SklearnEstimator.SklearnArgument']:
        """
        A set that contains the command line arguments that allow to control the hyperparameters of the estimator.
        """
        arguments: Set[SklearnEstimator.SklearnArgument] = set()
        apidoc = self.estimator_type.__doc__

        if apidoc:
            prefix = 'meta_' if self.is_meta_estimator else ''

            for param in docstring_parser.parse(apidoc).params:
                parameter_name = param.arg_name
                type_name = param.type_name

                if type_name and not parameter_name.startswith('_') and not parameter_name.endswith('_'):
                    argument_name = Argument.key_to_argument_name(prefix + parameter_name)
                    description = self.__format_argument_description(param.description or '')

                    try:
                        arguments.add(
                            SklearnEstimator.SklearnArgument.from_type_name(argument_name=argument_name,
                                                                            parameter_name=parameter_name,
                                                                            type_name=type_name,
                                                                            description=description))
                    except ValueError:
                        pass

        return arguments

    @cached_property
    def can_be_default_instantiated(self) -> bool:
        """
        True, if the estimator can be instantiated via a default constructor, False otherwise.
        """
        return self.__can_be_instantiated()

    @cached_property
    def can_be_instantiated_with_estimator(self) -> bool:
        """
        True, if the estimator can be instantiated by providing another estimator as a constructor argument, False
        otherwise.
        """
        return self.__can_be_instantiated(estimator=DummyRegressor() if self.is_regressor else DummyClassifier())

    def instantiate(self, args: Optional[Namespace] = None, **kwargs) -> SkLearnClassifierMixin | SkLearnRegressorMixin:
        """
        Creates and returns a new instance of the estimator.

        :param args:    Command line arguments specified by the user or None, if default hyperparameters should be used
        :param kwargs:  Optional keyword arguments to be passed to the estimator's constructor
        :return:        The instance that has been created
        """
        constructor_kwargs: Dict[str, Any] = dict(kwargs)

        if args:
            for argument in self.algorithmic_arguments:
                value = argument.get_value(args)

                if value is not None:
                    constructor_kwargs[argument.parameter_name] = value

        return self.estimator_type(**constructor_kwargs)

    @override
    def __str__(self) -> str:
        return self.estimator_name


class SkLearnEstimatorRunnable(SkLearnRunnable):
    """
    An abstract base class for all programs that run an experiment using a specific scikit-learn estimator.
    """

    class EstimatorExtension(Extension):
        """
        An extension that configures the scikit-learn estimator to be used in an experiment.
        """

        def __init__(self, supported_classifiers: Set[SklearnEstimator], supported_regressors: Set[SklearnEstimator],
                     supported_meta_classifiers: Set[SklearnEstimator],
                     supported_meta_regressors: Set[SklearnEstimator], *dependencies: 'Extension'):
            """
            :param supported_classifiers:       A set that contains all supported scikit-learn classifiers
            :param supported_regressors:        A set that contains all supported scikit-learn regressors
            :param supported_meta_classifiers:  A set that contains all supported scikit-learn meta-classifiers
            :param supported_meta_regressors:   A set that contains all supported scikit-learn meta-regressors
            :param dependencies:                Other extensions, this extension depends on
            """
            super().__init__(*dependencies)
            self._supported_classifiers = supported_classifiers
            self._supported_regressors = supported_regressors
            self._supported_meta_classifiers = supported_meta_classifiers
            self._supported_meta_regressors = supported_meta_regressors

        @staticmethod
        def create_meta_estimator_argument(supported_meta_classifiers: Set[SklearnEstimator],
                                           supported_meta_regressors: Set[SklearnEstimator]) -> SetArgument:
            """
            Creates and returns a `SetArgument` that allows to specify the name of a scikit-learn meta-estimator to be
            used in an experiment.

            :param supported_meta_classifiers:  A set that contains all supported scikit-learn meta-classifiers
            :param supported_meta_regressors:   A set that contains all supported scikit-learn meta-regressors
            :return:                            The `SetArgument` that has been created
            """
            return SetArgument(
                '--meta-estimator',
                values=set(map(str, chain(supported_meta_classifiers, supported_meta_regressors))),
                description='The name of a scikit-learn meta estimator to be used. Must be one of '
                + format_set(supported_meta_classifiers) + ', if the argument '
                + SkLearnRunnable.ProblemDomainExtension.PROBLEM_TYPE.name + ' is set to "' + ClassificationProblem.NAME
                + '", or ' + format_set(supported_meta_regressors) + ', if it is set to "' + RegressionProblem.NAME
                + '".',
                description_formatter=lambda description, _: description,
            )

        @staticmethod
        def create_estimator_argument(supported_classifiers: Set[SklearnEstimator],
                                      supported_regressors: Set[SklearnEstimator]) -> SetArgument:
            """
            Creates and returns a `SetArgument` that allows to specify the name of the scikit-learn estimator to be
            used in an experiment.

            :param supported_classifiers:   A set that contains all supported scikit-learn classifiers
            :param supported_regressors:    A set that contains all supported scikit-learn regressors
            :return                         The `SetArgument` that has been created
            """
            return SetArgument(
                '--estimator',
                values=set(map(str, chain(supported_classifiers, supported_regressors))),
                description='The name of the scikit-learn estimator to be used. Must be one of '
                + format_set(supported_classifiers) + ', if the argument '
                + SkLearnRunnable.ProblemDomainExtension.PROBLEM_TYPE.name + ' is set to "' + ClassificationProblem.NAME
                + '", or ' + format_set(supported_regressors) + ', if it is set to "' + RegressionProblem.NAME + '".',
                description_formatter=lambda description, _: description,
                required=True,
            )

        @override
        def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
            """
            return {
                self.create_meta_estimator_argument(supported_meta_classifiers=self._supported_meta_classifiers,
                                                    supported_meta_regressors=self._supported_meta_regressors),
                self.create_estimator_argument(supported_classifiers=self._supported_classifiers,
                                               supported_regressors=self._supported_regressors)
            }

        @override
        def get_supported_modes(self) -> Set[ExperimentMode]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
            """
            return {ExperimentMode.SINGLE, ExperimentMode.BATCH, ExperimentMode.RUN, ExperimentMode.READ}

    def __init__(self):
        self._classifiers = SklearnEstimator.get_supported_classifiers()
        self._regressors = SklearnEstimator.get_supported_regressors()
        self._meta_classifiers = SklearnEstimator.get_supported_meta_classifiers()
        self._meta_regressors = SklearnEstimator.get_supported_meta_regressors()

    @staticmethod
    def __get_estimator_by_name(estimators: Iterable[SklearnEstimator],
                                estimator_name: str,
                                problem_type: Optional[str] = None) -> Optional[SklearnEstimator]:
        estimators_by_name = {estimator.estimator_name: estimator for estimator in estimators}
        estimator = estimators_by_name.get(estimator_name)

        if not estimator and problem_type:
            raise ValueError('Estimator "' + estimator_name + '" does not support problem type "' + problem_type + '"')

        return estimator

    def __get_estimator(self, args: Namespace, problem_type: Optional[str] = None) -> Optional[SklearnEstimator]:
        estimators = self._regressors if problem_type == RegressionProblem.NAME else self._classifiers
        estimator_name = SkLearnEstimatorRunnable.EstimatorExtension.create_estimator_argument(
            supported_classifiers=self._classifiers, supported_regressors=self._regressors).get_value(args)
        return self.__get_estimator_by_name(estimators, estimator_name, problem_type=problem_type)

    def __get_meta_estimator(self, args: Namespace, problem_type: Optional[str] = None) -> Optional[SklearnEstimator]:
        meta_estimators = self._meta_regressors if problem_type == RegressionProblem.NAME else self._meta_classifiers
        meta_estimator_name = SkLearnEstimatorRunnable.EstimatorExtension.create_meta_estimator_argument(
            supported_meta_classifiers=self._meta_classifiers,
            supported_meta_regressors=self._meta_regressors).get_value(args)

        if meta_estimator_name:
            return self.__get_estimator_by_name(meta_estimators, meta_estimator_name, problem_type=problem_type)

        return None

    def __instantiate_estimator(
            self,
            args: Namespace,
            problem_type: Optional[str] = None) -> Optional[SkLearnClassifierMixin] | Optional[SkLearnRegressorMixin]:
        estimator = self.__get_estimator(args, problem_type=problem_type)
        estimator = estimator.instantiate(args) if estimator else None

        if estimator is not None:
            meta_estimator = self.__get_meta_estimator(args, problem_type=problem_type)

            if meta_estimator is not None:
                return meta_estimator.instantiate(args, estimator=estimator)

        return estimator

    @override
    def get_extensions(self) -> List[Extension]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_extensions`
        """
        return [
            SkLearnEstimatorRunnable.EstimatorExtension(supported_classifiers=self._classifiers,
                                                        supported_regressors=self._regressors,
                                                        supported_meta_classifiers=self._meta_classifiers,
                                                        supported_meta_regressors=self._meta_regressors),
        ] + super().get_extensions()

    @override
    def get_algorithmic_arguments(self, known_args: Namespace) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_algorithmic_arguments`
        """
        meta_estimator = self.__get_meta_estimator(known_args)
        meta_estimator_arguments = meta_estimator.algorithmic_arguments if meta_estimator else set()
        estimator = self.__get_estimator(known_args)
        estimator_arguments = estimator.algorithmic_arguments if estimator else set()
        return meta_estimator_arguments | estimator_arguments

    @override
    def create_classifier(self, args: Namespace) -> Optional[SkLearnClassifierMixin]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.create_classifier`
        """
        return self.__instantiate_estimator(args, problem_type=ClassificationProblem.NAME)

    @override
    def create_regressor(self, args: Namespace) -> Optional[SkLearnRegressorMixin]:
        """
        See :func:`mlrl.testbed_sklearn.runnables.SkLearnRunnable.create_regressor`
        """
        return self.__instantiate_estimator(args, problem_type=RegressionProblem.NAME)
