"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for running experiments using the scikit-learn framework.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Dict, Optional, Set

from sklearn.base import ClassifierMixin as SkLearnClassifierMixin, RegressorMixin as SkLearnRegressorMixin

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

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter
from mlrl.testbed.experiments.input.model.extension import ModelInputExtension
from mlrl.testbed.experiments.input.parameters.extension import ParameterInputExtension
from mlrl.testbed.experiments.output.model.extension import ModelOutputExtension
from mlrl.testbed.experiments.output.parameters.extension import ParameterOutputExtension
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain, RegressionProblem
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.runnables import Runnable

from mlrl.util.cli import Argument, SetArgument


class SkLearnRunnable(Runnable, ABC):
    """
    An abstract base class for all programs that run an experiment using the scikit-learn framework.
    """

    class GlobalPredictorFactory(SkLearnProblem.PredictorFactory):
        """
        Allow to create instances of type `Predictor` that obtain predictions from a global model.
        """

        def __init__(self, prediction_type: PredictionType):
            """
            :param prediction_type: The type of the predictions to be obtained
            """
            self.prediction_type = prediction_type

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

        def _get_arguments(self) -> Set[Argument]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
            """
            return {self.PROBLEM_TYPE}

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
                return SkLearnClassificationProblem(base_learner=runnable.create_classifier(args),
                                                    predictor_factory=predictor_factory,
                                                    prediction_type=prediction_type,
                                                    fit_kwargs=fit_kwargs,
                                                    predict_kwargs=predict_kwargs)

            return SkLearnRegressionProblem(base_learner=runnable.create_regressor(args),
                                            predictor_factory=predictor_factory,
                                            prediction_type=prediction_type,
                                            fit_kwargs=fit_kwargs,
                                            predict_kwargs=predict_kwargs)

    def get_extensions(self) -> Set[Extension]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_extensions`
        """
        return super().get_extensions() | {
            SkLearnRunnable.ProblemDomainExtension(),
            DatasetSplitterExtension(),
            PredictionTypeExtension(),
            ModelInputExtension(),
            ModelOutputExtension(),
            ParameterInputExtension(),
            ParameterOutputExtension(),
            EvaluationExtension(),
            TabularDataCharacteristicExtension(),
            LabelVectorExtension(),
            PredictionExtension(),
            GroundTruthExtension(),
            PredictionCharacteristicsExtension(),
        }

    def create_experiment_builder(self, args: Namespace) -> Experiment.Builder:
        """
        See :func:`mlrl.testbed.runnables.Runnable.create_experiment_builder`
        """
        dataset_splitter = DatasetSplitterExtension.get_dataset_splitter(args)
        return self._create_experiment_builder(args, dataset_splitter)

    def _create_experiment_builder(self, args: Namespace, dataset_splitter: DatasetSplitter) -> Experiment.Builder:
        """
        May be overridden by subclasses in order to create the `Experiment` that should be run.

        :param args:                The command line arguments
        :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
        :return:                    The `Experiment` that has been created
        """
        problem_domain = SkLearnRunnable.ProblemDomainExtension.get_problem_domain(args, runnable=self)
        return SkLearnExperiment.Builder(problem_domain=problem_domain, dataset_splitter=dataset_splitter)

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
