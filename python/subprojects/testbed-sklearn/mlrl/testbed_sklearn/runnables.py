"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for running experiments using the scikit-learn framework.
"""
import contextlib
import os

from abc import ABC, abstractmethod
from argparse import Namespace
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, override

import numpy as np

from sklearn.base import ClassifierMixin as SkLearnClassifierMixin, RegressorMixin as SkLearnRegressorMixin
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

from mlrl.util.cli import Argument, SetArgument
from mlrl.util.format import format_set


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

    EstimatorType = Type[SkLearnClassifierMixin] | Type[SkLearnRegressorMixin]

    def __init__(self, estimator_name: str, estimator_type: EstimatorType):
        """
        :param estimator_name:  The name of the estimator
        :param estimator_type:  The type of the estimator
        """
        self.estimator_name = estimator_name
        self.estimator_type = estimator_type

    @staticmethod
    def get_supported_estimators(problem_type: Optional[str] = None) -> Set['SklearnEstimator']:
        """
        Returns a set that returns all supported scikit-learn estimators for a given problem type.

        :param problem_type:    A problem type or None, if all estimators should be returned
        :return:                A set that contains the names of all supported estimators
        """
        regressors: Set[SklearnEstimator] = set()
        classifiers: Set[SklearnEstimator] = set()

        if not problem_type or problem_type == RegressionProblem.NAME:
            regressors = {
                SklearnEstimator(estimator_name=estimator_name, estimator_type=estimator_type)
                for estimator_name, estimator_type in all_estimators(type_filter='regressor')
                if issubclass(estimator_type, SkLearnRegressorMixin)
            }

        if not problem_type or problem_type == ClassificationProblem.NAME:
            classifiers = {
                SklearnEstimator(estimator_name=estimator_name, estimator_type=estimator_type)
                for estimator_name, estimator_type in all_estimators(type_filter='classifier')
                if issubclass(estimator_type, SkLearnClassifierMixin)
            }

        return set(filter(lambda estimator: estimator.can_be_default_instantiated, chain(regressors, classifiers)))

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

    @cached_property
    def can_be_default_instantiated(self) -> bool:
        """
        True, if the estimator can be instantiated via a default constructor, False otherwise.
        """

        @contextlib.contextmanager
        def suppress_output():
            with open(os.devnull, mode='w', encoding=ENCODING_UTF8) as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    yield

        try:
            with suppress_output():
                instance = self.instantiate()
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

    def instantiate(self) -> SkLearnClassifierMixin | SkLearnRegressorMixin:
        """
        Creates and returns a new instance of the estimator.

        :return: The instance that has been created
        """
        return self.estimator_type()

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

        ESTIMATOR = SetArgument(
            '--estimator',
            values=set(map(str, SklearnEstimator.get_supported_estimators())),
            description='The name of the scikit-learn estimator to be used. Must be one of '
            + format_set(SklearnEstimator.get_supported_estimators(ClassificationProblem.NAME)) + ', if the argument '
            + SkLearnRunnable.ProblemDomainExtension.PROBLEM_TYPE.name + ' is set to "' + ClassificationProblem.NAME
            + '", or ' + format_set(SklearnEstimator.get_supported_estimators(RegressionProblem.NAME)) + ', if it is '
            + 'set to "' + RegressionProblem.NAME + '".',
            description_formatter=lambda description, _: description,
            required=True,
        )

        @override
        def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
            """
            return {self.ESTIMATOR}

        @override
        def get_supported_modes(self) -> Set[ExperimentMode]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
            """
            return {ExperimentMode.SINGLE, ExperimentMode.BATCH, ExperimentMode.RUN, ExperimentMode.READ}

    @staticmethod
    def __get_estimator(args: Namespace, problem_type: str) -> SklearnEstimator:
        estimator_name = SkLearnEstimatorRunnable.EstimatorExtension.ESTIMATOR.get_value(args)
        estimators_by_name = {
            estimator.estimator_name: estimator
            for estimator in SklearnEstimator.get_supported_estimators(problem_type=problem_type)
        }
        estimator = estimators_by_name.get(estimator_name)

        if not estimator:
            raise ValueError('Estimator "' + estimator_name + '" does not support problem type "' + problem_type + '"')

        return estimator

    @override
    def get_extensions(self) -> List[Extension]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_extensions`
        """
        return [
            SkLearnEstimatorRunnable.EstimatorExtension(),
        ] + super().get_extensions()

    @override
    def get_algorithmic_arguments(self, known_args: Namespace) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_algorithmic_arguments`
        """
        return set()

    @override
    def create_classifier(self, args: Namespace) -> Optional[SkLearnClassifierMixin]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.create_classifier`
        """
        estimator = self.__get_estimator(args, problem_type=ClassificationProblem.NAME)
        return estimator.instantiate()

    @override
    def create_regressor(self, args: Namespace) -> Optional[SkLearnRegressorMixin]:
        """
        See :func:`mlrl.testbed_sklearn.runnables.SkLearnRunnable.create_regressor`
        """
        estimator = self.__get_estimator(args, problem_type=RegressionProblem.NAME)
        return estimator.instantiate()
