"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for running experiments using the scikit-learn framework.
"""

from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set

from sklearn.base import ClassifierMixin as SkLearnClassifierMixin, RegressorMixin as SkLearnRegressorMixin

from mlrl.common.config.parameters import NONE

from mlrl.testbed_arff.experiments.input.sources import ArffFileSource

from mlrl.testbed.experiments import Experiment, SkLearnExperiment
from mlrl.testbed.experiments.input.dataset import DatasetReader, InputDataset
from mlrl.testbed.experiments.input.dataset.preprocessors import Preprocessor
from mlrl.testbed.experiments.input.dataset.preprocessors.tabular import OneHotEncoder
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter, NoSplitter
from mlrl.testbed.experiments.input.dataset.splitters.tabular import BipartitionSplitter, CrossValidationSplitter
from mlrl.testbed.experiments.input.model.extension import ModelInputExtension
from mlrl.testbed.experiments.input.parameters.extension import ParameterInputExtension
from mlrl.testbed.experiments.output.characteristics.data.tabular.extension import TabularDataCharacteristicExtension
from mlrl.testbed.experiments.output.characteristics.data.tabular.extension_prediction import \
    PredictionCharacteristicsExtension
from mlrl.testbed.experiments.output.dataset.tabular.extension_ground_truth import GroundTruthExtension
from mlrl.testbed.experiments.output.dataset.tabular.extension_prediction import PredictionExtension
from mlrl.testbed.experiments.output.evaluation.extension import EvaluationExtension
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.label_vectors.extension import LabelVectorExtension
from mlrl.testbed.experiments.output.model.extension import ModelOutputExtension
from mlrl.testbed.experiments.output.parameters.extension import ParameterOutputExtension
from mlrl.testbed.experiments.prediction import GlobalPredictor
from mlrl.testbed.experiments.prediction.extension import PredictionTypeExtension
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain, RegressionProblem
from mlrl.testbed.experiments.problem_domain_sklearn import SkLearnClassificationProblem, SkLearnProblem, \
    SkLearnRegressionProblem
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.runnables import Runnable

from mlrl.util.cli import BoolArgument, CommandLineInterface, IntArgument, SetArgument, StringArgument
from mlrl.util.format import format_enum_values, format_set
from mlrl.util.options import BooleanOption, parse_param, parse_param_and_options
from mlrl.util.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal


class SkLearnRunnable(Runnable, ABC):
    """
    An abstract base class for all programs that run an experiment using the scikit-learn framework.
    """

    PARAM_PROBLEM_TYPE = '--problem-type'

    PROBLEM_TYPE_VALUES = {
        ClassificationProblem.NAME,
        RegressionProblem.NAME,
    }

    PARAM_RANDOM_STATE = '--random-state'

    PARAM_DATA_SPLIT = '--data-split'

    DATA_SPLIT_TRAIN_TEST = 'train-test'

    OPTION_TEST_SIZE = 'test_size'

    DATA_SPLIT_CROSS_VALIDATION = 'cross-validation'

    OPTION_NUM_FOLDS = 'num_folds'

    OPTION_FIRST_FOLD = 'last_fold'

    OPTION_LAST_FOLD = 'first_fold'

    DATA_SPLIT_VALUES: Dict[str, Set[str]] = {
        NONE: {},
        DATA_SPLIT_TRAIN_TEST: {OPTION_TEST_SIZE},
        DATA_SPLIT_CROSS_VALIDATION: {OPTION_NUM_FOLDS, OPTION_FIRST_FOLD, OPTION_LAST_FOLD}
    }

    def _create_problem_domain(self,
                               args: Namespace,
                               fit_kwargs: Optional[Dict[str, Any]] = None,
                               predict_kwargs: Optional[Dict[str, Any]] = None) -> ProblemDomain:
        prediction_type = PredictionTypeExtension.get_prediction_type(args)
        predictor_factory = self._create_predictor_factory(args, prediction_type)
        value = parse_param(self.PARAM_PROBLEM_TYPE, args.problem_type, self.PROBLEM_TYPE_VALUES)

        if value == ClassificationProblem.NAME:
            base_learner = self.create_classifier(args)
            return SkLearnClassificationProblem(base_learner=base_learner,
                                                predictor_factory=predictor_factory,
                                                prediction_type=prediction_type,
                                                fit_kwargs=fit_kwargs,
                                                predict_kwargs=predict_kwargs)

        base_learner = self.create_regressor(args)
        return SkLearnRegressionProblem(base_learner=base_learner,
                                        predictor_factory=predictor_factory,
                                        prediction_type=prediction_type,
                                        fit_kwargs=fit_kwargs,
                                        predict_kwargs=predict_kwargs)

    @staticmethod
    def __create_preprocessors(args) -> List[Preprocessor]:
        preprocessors = []

        if args.one_hot_encoding:
            preprocessors.append(OneHotEncoder())

        return preprocessors

    def __create_dataset_splitter(self, args) -> DatasetSplitter:
        dataset = InputDataset(name=args.dataset)
        source = ArffFileSource(directory=args.data_dir)
        dataset_reader = DatasetReader(source=source, input_data=dataset)
        dataset_reader.add_preprocessors(*self.__create_preprocessors(args))
        value, options = parse_param_and_options(self.PARAM_DATA_SPLIT, args.data_split, self.DATA_SPLIT_VALUES)

        if value == self.DATA_SPLIT_CROSS_VALIDATION:
            num_folds = options.get_int(self.OPTION_NUM_FOLDS, 10)
            assert_greater_or_equal(self.OPTION_NUM_FOLDS, num_folds, 2)
            first_fold = options.get_int(self.OPTION_FIRST_FOLD, 1)
            assert_greater_or_equal(self.OPTION_FIRST_FOLD, first_fold, 1)
            assert_less_or_equal(self.OPTION_FIRST_FOLD, first_fold, num_folds)
            last_fold = options.get_int(self.OPTION_LAST_FOLD, num_folds)
            assert_greater_or_equal(self.OPTION_LAST_FOLD, last_fold, first_fold)
            assert_less_or_equal(self.OPTION_LAST_FOLD, last_fold, num_folds)
            random_state = int(args.random_state) if args.random_state else 1
            assert_greater_or_equal(self.PARAM_RANDOM_STATE, random_state, 1)
            return CrossValidationSplitter(dataset_reader,
                                           num_folds=num_folds,
                                           first_fold=first_fold - 1,
                                           last_fold=last_fold,
                                           random_state=random_state)
        if value == self.DATA_SPLIT_TRAIN_TEST:
            test_size = options.get_float(self.OPTION_TEST_SIZE, 0.33)
            assert_greater(self.OPTION_TEST_SIZE, test_size, 0)
            assert_less(self.OPTION_TEST_SIZE, test_size, 1)
            random_state = int(args.random_state) if args.random_state else 1
            assert_greater_or_equal(self.PARAM_RANDOM_STATE, random_state, 1)
            return BipartitionSplitter(dataset_reader, test_size=test_size, random_state=random_state)

        return NoSplitter(dataset_reader)

    def get_extensions(self) -> List[Extension]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_extensions`
        """
        return super().get_extensions() + [
            PredictionTypeExtension(),
            OutputExtension(),
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
        ]

    def configure_arguments(self, cli: CommandLineInterface):
        """
        See :func:`mlrl.testbed.runnables.Runnable.configure_arguments`
        """
        super().configure_arguments(cli)
        cli.add_arguments(
            SetArgument(
                self.PARAM_PROBLEM_TYPE,
                values={ClassificationProblem.NAME, RegressionProblem.NAME},
                default=ClassificationProblem.NAME,
                help='The type of the machine learning problem to be solved. Must be one of '
                + format_set(self.PROBLEM_TYPE_VALUES) + '.',
            ),
            IntArgument(
                self.PARAM_RANDOM_STATE,
                help='The seed to be used by random number generators. Must be at least 1.',
            ),
            StringArgument(
                '--data-dir',
                required=True,
                help='The path to the directory where the data set files are located.',
            ),
            StringArgument(
                '--dataset',
                required=True,
                help='The name of the data set files without suffix.',
            ),
            SetArgument(
                self.PARAM_DATA_SPLIT,
                values=self.DATA_SPLIT_VALUES,
                default=self.DATA_SPLIT_TRAIN_TEST,
                help='The strategy to be used for splitting the available data into training and test sets.',
            ),
            BoolArgument(
                '--one-hot-encoding',
                default=False,
                help='Whether one-hot-encoding should be used to encode nominal features or not.',
            ),
            BoolArgument(
                '--create-output-dir',
                default=True,
                help='Whether the directories specified via the arguments ' + OutputExtension.OUTPUT_DIR.name + ', '
                + ModelOutputExtension.MODEL_SAVE_DIR.name + ' and ' + ParameterOutputExtension.PARAMETER_SAVE_DIR.name
                + ' should automatically be created, if they do not exist, or not. Must be one of '
                + format_enum_values(BooleanOption) + '.',
            ),
            BoolArgument(
                '--print-all',
                default=False,
                help='Whether all output data should be printed on the console or not.',
            ),
            BoolArgument(
                '--store-all',
                default=False,
                help='Whether all output data should be written to files or not.',
            ),
        )

    def create_experiment_builder(self, args: Namespace) -> Experiment.Builder:
        """
        See :func:`mlrl.testbed.runnables.Runnable.create_experiment_builder`
        """
        dataset_splitter = self.__create_dataset_splitter(args)
        return self._create_experiment_builder(args, dataset_splitter)

    def _create_experiment_builder(self, args: Namespace, dataset_splitter: DatasetSplitter) -> Experiment.Builder:
        """
        May be overridden by subclasses in order to create the `Experiment` that should be run.

        :param args:                The command line arguments
        :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
        :return:                    The `Experiment` that has been created
        """
        problem_domain = self._create_problem_domain(args)
        return SkLearnExperiment.Builder(problem_domain=problem_domain, dataset_splitter=dataset_splitter)

    # pylint: disable=unused-argument
    def _create_predictor_factory(self, args, prediction_type: PredictionType) -> SkLearnProblem.PredictorFactory:
        """
        May be overridden by subclasses in order to create the `SkLearnProblem.PredictorFactory` that should be used for
        obtaining predictions from a previously trained model.

        :param args:            The command line arguments
        :param prediction_type: The type of the predictions to be obtained
        :return:                The `SkLearnProblem.PredictorFactory` that has been created
        """

        def predictor_factory():
            return GlobalPredictor(prediction_type)

        return predictor_factory

    @abstractmethod
    def create_classifier(self, args) -> Optional[SkLearnClassifierMixin]:
        """
        Must be implemented by subclasses in order to create a machine learning algorithm that can be applied to
        classification problems.

        :param args:    The command line arguments
        :return:        The learner that has been created or None, if regression problems are not supported
        """

    @abstractmethod
    def create_regressor(self, args) -> Optional[SkLearnRegressorMixin]:
        """
        Must be implemented by subclasses in order to create a machine learning algorithm that can be applied to
        regression problems.

        :param args:    The command line arguments
        :return:        The learner that has been created or None, if regression problems are not supported
        """
