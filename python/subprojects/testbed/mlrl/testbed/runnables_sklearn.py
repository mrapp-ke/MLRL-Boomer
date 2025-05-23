"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for running experiments using the scikit-learn framework.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentError, ArgumentParser, Namespace
from os import listdir, path, unlink
from typing import Any, Dict, List, Optional, Set

from sklearn.base import ClassifierMixin as SkLearnClassifierMixin, RegressorMixin as SkLearnRegressorMixin

from mlrl.common.config.parameters import AUTOMATIC, NONE, Parameter
from mlrl.common.learners import RuleLearner, SparsePolicy

from mlrl.testbed_arff.experiments.input.sources import ArffFileSource
from mlrl.testbed_arff.experiments.output.sinks import ArffFileSink

from mlrl.testbed.experiments import Experiment, SkLearnExperiment
from mlrl.testbed.experiments.input.dataset import DatasetReader, InputDataset
from mlrl.testbed.experiments.input.dataset.preprocessors import Preprocessor
from mlrl.testbed.experiments.input.dataset.preprocessors.tabular import OneHotEncoder
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter, NoSplitter
from mlrl.testbed.experiments.input.dataset.splitters.tabular import BipartitionSplitter, CrossValidationSplitter
from mlrl.testbed.experiments.input.model import ModelReader
from mlrl.testbed.experiments.input.parameters import ParameterReader
from mlrl.testbed.experiments.input.sources import CsvFileSource, PickleFileSource
from mlrl.testbed.experiments.output.characteristics.data.tabular import DataCharacteristics, \
    DataCharacteristicsWriter, OutputCharacteristics, PredictionCharacteristicsWriter
from mlrl.testbed.experiments.output.characteristics.model import ModelCharacteristicsWriter, \
    RuleModelCharacteristicsExtractor
from mlrl.testbed.experiments.output.dataset.tabular import GroundTruthWriter, PredictionWriter
from mlrl.testbed.experiments.output.evaluation import ClassificationEvaluationDataExtractor, EvaluationResult, \
    EvaluationWriter, RankingEvaluationDataExtractor, RegressionEvaluationDataExtractor
from mlrl.testbed.experiments.output.label_vectors import LabelVectors, LabelVectorSetExtractor, LabelVectorWriter
from mlrl.testbed.experiments.output.model import ModelWriter
from mlrl.testbed.experiments.output.model_text import ModelAsTextWriter, RuleModelAsText, RuleModelAsTextExtractor
from mlrl.testbed.experiments.output.parameters import ParameterWriter
from mlrl.testbed.experiments.output.probability_calibration import IsotonicJointProbabilityCalibrationModelExtractor, \
    IsotonicMarginalProbabilityCalibrationModelExtractor, ProbabilityCalibrationModelWriter
from mlrl.testbed.experiments.output.sinks import CsvFileSink, LogSink, PickleFileSink, TextFileSink
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.prediction import GlobalPredictor, IncrementalPredictor
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain, RegressionProblem
from mlrl.testbed.experiments.problem_domain_sklearn import SkLearnClassificationProblem, SkLearnProblem, \
    SkLearnRegressionProblem
from mlrl.testbed.runnables import Runnable
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.format import format_enum_values, format_iterable, format_set
from mlrl.util.options import BooleanOption, parse_param, parse_param_and_options
from mlrl.util.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal


class SkLearnRunnable(Runnable, ABC):
    """
    An abstract base class for all programs that run an experiment using the scikit-learn framework.
    """

    class ClearOutputDirectoryListener(Experiment.Listener):
        """
        Deletes all files from the output directory before an experiment starts.
        """

        def __init__(self, output_dir: str):
            """
            :param output_dir: The path to the output directory from which the files should be deleted
            """
            self.output_dir = output_dir

        def before_start(self, _: Experiment):
            """
            See :func:`mlrl.testbed.experiments.Experiment.Listener.before_start`
            """
            output_dir = self.output_dir

            if path.isdir(output_dir):
                for file in listdir(output_dir):
                    file_path = path.join(output_dir, file)

                    if path.isfile(file_path):
                        unlink(file_path)

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

    PARAM_PRINT_EVALUATION = '--print-evaluation'

    PRINT_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            EvaluationResult.OPTION_ENABLE_ALL, EvaluationResult.OPTION_HAMMING_LOSS,
            EvaluationResult.OPTION_HAMMING_ACCURACY, EvaluationResult.OPTION_SUBSET_ZERO_ONE_LOSS,
            EvaluationResult.OPTION_SUBSET_ACCURACY, EvaluationResult.OPTION_MICRO_PRECISION,
            EvaluationResult.OPTION_MICRO_RECALL, EvaluationResult.OPTION_MICRO_F1,
            EvaluationResult.OPTION_MICRO_JACCARD, EvaluationResult.OPTION_MACRO_PRECISION,
            EvaluationResult.OPTION_MACRO_RECALL, EvaluationResult.OPTION_MACRO_F1,
            EvaluationResult.OPTION_MACRO_JACCARD, EvaluationResult.OPTION_EXAMPLE_WISE_PRECISION,
            EvaluationResult.OPTION_EXAMPLE_WISE_RECALL, EvaluationResult.OPTION_EXAMPLE_WISE_F1,
            EvaluationResult.OPTION_EXAMPLE_WISE_JACCARD, EvaluationResult.OPTION_ACCURACY,
            EvaluationResult.OPTION_ZERO_ONE_LOSS, EvaluationResult.OPTION_PRECISION, EvaluationResult.OPTION_RECALL,
            EvaluationResult.OPTION_F1, EvaluationResult.OPTION_JACCARD, EvaluationResult.OPTION_MEAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_SQUARED_ERROR, EvaluationResult.OPTION_MEDIAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, EvaluationResult.OPTION_RANK_LOSS,
            EvaluationResult.OPTION_COVERAGE_ERROR, EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
            EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_EVALUATION = '--store-evaluation'

    STORE_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            EvaluationResult.OPTION_ENABLE_ALL, EvaluationResult.OPTION_HAMMING_LOSS,
            EvaluationResult.OPTION_HAMMING_ACCURACY, EvaluationResult.OPTION_SUBSET_ZERO_ONE_LOSS,
            EvaluationResult.OPTION_SUBSET_ACCURACY, EvaluationResult.OPTION_MICRO_PRECISION,
            EvaluationResult.OPTION_MICRO_RECALL, EvaluationResult.OPTION_MICRO_F1,
            EvaluationResult.OPTION_MICRO_JACCARD, EvaluationResult.OPTION_MACRO_PRECISION,
            EvaluationResult.OPTION_MACRO_RECALL, EvaluationResult.OPTION_MACRO_F1,
            EvaluationResult.OPTION_MACRO_JACCARD, EvaluationResult.OPTION_EXAMPLE_WISE_PRECISION,
            EvaluationResult.OPTION_EXAMPLE_WISE_RECALL, EvaluationResult.OPTION_EXAMPLE_WISE_F1,
            EvaluationResult.OPTION_EXAMPLE_WISE_JACCARD, EvaluationResult.OPTION_ACCURACY,
            EvaluationResult.OPTION_ZERO_ONE_LOSS, EvaluationResult.OPTION_PRECISION, EvaluationResult.OPTION_RECALL,
            EvaluationResult.OPTION_F1, EvaluationResult.OPTION_JACCARD, EvaluationResult.OPTION_MEAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_SQUARED_ERROR, EvaluationResult.OPTION_MEDIAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, EvaluationResult.OPTION_RANK_LOSS,
            EvaluationResult.OPTION_COVERAGE_ERROR, EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
            EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, EvaluationResult.OPTION_TRAINING_TIME,
            EvaluationResult.OPTION_PREDICTION_TIME, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_PRINT_PREDICTIONS = '--print-predictions'

    PRINT_PREDICTIONS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_PREDICTIONS = '--store-predictions'

    STORE_PREDICTIONS_VALUES = PRINT_PREDICTIONS_VALUES

    PARAM_PRINT_GROUND_TRUTH = '--print-ground-truth'

    PRINT_GROUND_TRUTH_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_GROUND_TRUTH = '--store-ground-truth'

    STORE_GROUND_TRUTH_VALUES = PRINT_GROUND_TRUTH_VALUES

    PARAM_PRINT_PREDICTION_CHARACTERISTICS = '--print-prediction-characteristics'

    PRINT_PREDICTION_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
            OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
            OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
            OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_PREDICTION_CHARACTERISTICS = '--store-prediction-characteristics'

    STORE_PREDICTION_CHARACTERISTICS_VALUES = PRINT_PREDICTION_CHARACTERISTICS_VALUES

    PARAM_PRINT_DATA_CHARACTERISTICS = '--print-data-characteristics'

    PRINT_DATA_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            DataCharacteristics.OPTION_EXAMPLES, DataCharacteristics.OPTION_FEATURES,
            DataCharacteristics.OPTION_NUMERICAL_FEATURES, DataCharacteristics.OPTION_NOMINAL_FEATURES,
            DataCharacteristics.OPTION_FEATURE_DENSITY, DataCharacteristics.OPTION_FEATURE_SPARSITY,
            OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
            OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
            OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
            OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_DATA_CHARACTERISTICS = '--store-data-characteristics'

    STORE_DATA_CHARACTERISTICS_VALUES = PRINT_DATA_CHARACTERISTICS_VALUES

    PARAM_PRINT_LABEL_VECTORS = '--print-label-vectors'

    PRINT_LABEL_VECTORS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {LabelVectors.OPTION_SPARSE},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_LABEL_VECTORS = '--store-label-vectors'

    STORE_LABEL_VECTORS_VALUES = PRINT_LABEL_VECTORS_VALUES

    PARAM_MODEL_SAVE_DIR = '--model-save-dir'

    PARAM_PARAMETER_SAVE_DIR = '--parameter-save-dir'

    PARAM_OUTPUT_DIR = '--output-dir'

    PARAM_WIPE_OUTPUT_DIR = '--wipe-output-dir'

    WIPE_OUTPUT_DIR_VALUES = {
        BooleanOption.TRUE.value,
        BooleanOption.FALSE.value,
        AUTOMATIC,
    }

    PARAM_PREDICTION_TYPE = '--prediction-type'

    def _create_problem_domain(self,
                               args,
                               fit_kwargs: Optional[Dict[str, Any]] = None,
                               predict_kwargs: Optional[Dict[str, Any]] = None) -> ProblemDomain:
        prediction_type = self.__create_prediction_type(args)
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

    def __create_prediction_type(self, args) -> PredictionType:
        return PredictionType.parse(self.PARAM_PREDICTION_TYPE, args.prediction_type)

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

    def __create_clear_output_directory_listener(self, args,
                                                 dataset_splitter: DatasetSplitter) -> Optional[Experiment.Listener]:
        output_dir = args.output_dir

        if output_dir:
            value = parse_param(self.PARAM_WIPE_OUTPUT_DIR, args.wipe_output_dir, self.WIPE_OUTPUT_DIR_VALUES)

            if value == BooleanOption.TRUE.value or (value == AUTOMATIC
                                                     and not dataset_splitter.folding_strategy.is_subset):
                return SkLearnRunnable.ClearOutputDirectoryListener(output_dir)

        return None

    def configure_arguments(self, parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.runnables.Runnable.configure_arguments`
        """
        super().configure_arguments(parser)
        parser.add_argument(self.PARAM_PROBLEM_TYPE,
                            type=str,
                            default=ClassificationProblem.NAME,
                            help='The type of the machine learning problem to be solved. Must be one of '
                            + format_set(self.PROBLEM_TYPE_VALUES) + '.')
        parser.add_argument(self.PARAM_RANDOM_STATE,
                            type=int,
                            default=None,
                            help='The seed to be used by random number generators. Must be at least 1.')
        parser.add_argument('--data-dir',
                            type=str,
                            required=True,
                            help='The path to the directory where the data set files are located.')
        parser.add_argument('--dataset', type=str, required=True, help='The name of the data set files without suffix.')
        parser.add_argument(self.PARAM_DATA_SPLIT,
                            type=str,
                            default=self.DATA_SPLIT_TRAIN_TEST,
                            help='The strategy to be used for splitting the available data into training and test '
                            + 'sets. Must be one of ' + format_set(self.DATA_SPLIT_VALUES.keys()) + '. For additional '
                            + 'options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_EVALUATION,
                            type=str,
                            default=BooleanOption.TRUE.value,
                            help='Whether the evaluation results should be printed on the console or not. Must be one '
                            + 'of ' + format_set(self.PRINT_EVALUATION_VALUES.keys()) + '. For additional options '
                            + 'refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_EVALUATION,
                            type=str,
                            default=BooleanOption.TRUE.value,
                            help='Whether the evaluation results should be written into output files or not. Must be '
                            + 'one of ' + format_set(self.STORE_EVALUATION_VALUES.keys()) + '. Does only have an '
                            + 'effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional '
                            + 'options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_PREDICTION_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of binary predictions should be printed on the console '
                            + 'or not. Must be one of '
                            + format_set(self.PRINT_PREDICTION_CHARACTERISTICS_VALUES.keys()) + '. Does only have an '
                            + 'effect if the parameter ' + self.PARAM_PREDICTION_TYPE + ' is set to '
                            + PredictionType.BINARY.value + '. For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_PREDICTION_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of binary predictions should be written into output '
                            + 'files or not. Must be one of '
                            + format_set(self.STORE_PREDICTION_CHARACTERISTICS_VALUES.keys()) + '. Does only have an '
                            + 'effect if the parameter ' + self.PARAM_PREDICTION_TYPE + ' is set to '
                            + PredictionType.BINARY.value + '. For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_DATA_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of the training data should be printed on the console or '
                            + 'not. Must be one of ' + format_set(self.PRINT_DATA_CHARACTERISTICS_VALUES.keys()) + '. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_DATA_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of the training data should be written into output files '
                            + 'or not. Must be one of ' + format_set(self.STORE_DATA_CHARACTERISTICS_VALUES.keys())
                            + '. Does only have an effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_LABEL_VECTORS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the unique label vectors contained in the training data should be printed on '
                            + 'the console or not. Must be one of ' + format_set(self.PRINT_LABEL_VECTORS_VALUES.keys())
                            + '. For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_LABEL_VECTORS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the unique label vectors contained in the training data should be written '
                            + 'into output files or not. Must be one of '
                            + format_set(self.STORE_LABEL_VECTORS_VALUES.keys()) + '. Does only have an effect if the '
                            + 'parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to '
                            + 'the documentation.')
        parser.add_argument('--one-hot-encoding',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether one-hot-encoding should be used to encode nominal features or not. Must be '
                            + 'one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--model-load-dir',
                            type=str,
                            help='The path to the directory from which models should be loaded.')
        parser.add_argument(self.PARAM_MODEL_SAVE_DIR,
                            type=str,
                            help='The path to the directory to which models should be saved.')
        parser.add_argument('--parameter-load-dir',
                            type=str,
                            help='The path to the directory from which parameter to be used by the algorith should be '
                            + 'loaded.')
        parser.add_argument(self.PARAM_PARAMETER_SAVE_DIR,
                            type=str,
                            help='The path to the directory where configuration files, which specify the parameters to '
                            + 'be used by the algorithm, are located.')
        parser.add_argument(self.PARAM_OUTPUT_DIR,
                            type=str,
                            help='The path to the directory where experimental results should be saved.')
        parser.add_argument('--create-output-dir',
                            type=BooleanOption.parse,
                            default=True,
                            help='Whether the directories specified via the arguments ' + self.PARAM_OUTPUT_DIR + ', '
                            + self.PARAM_MODEL_SAVE_DIR + ' and ' + self.PARAM_PARAMETER_SAVE_DIR + ' should '
                            + 'automatically be created, if they do not exist, or not. Must be one of '
                            + format_enum_values(BooleanOption) + '.')
        parser.add_argument(self.PARAM_WIPE_OUTPUT_DIR,
                            type=str,
                            default=AUTOMATIC,
                            help='Whether all files in the directory specified via the argument '
                            + self.PARAM_OUTPUT_DIR + ' should be deleted before an experiment starts or not. Must be '
                            + 'one of ' + format_iterable(self.WIPE_OUTPUT_DIR_VALUES) + '. If set to ' + AUTOMATIC
                            + ', the files are only deleted if the experiment does not run a subset of the folds of a '
                            + 'cross validation.')
        parser.add_argument('--exit-on-error',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the program should exit if an error occurs while writing experimental '
                            + 'results or not. Must be one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--print-all',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether all output data should be printed on the console or not. Must be one of '
                            + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--store-all',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether all output data should be written to files or not. Must be one of '
                            + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--print-parameters',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the parameter setting should be printed on the console or not. Must be one '
                            + 'of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument(self.PARAM_PRINT_PREDICTIONS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether predictions should be printed on the console or not. Must be one of '
                            + format_set(self.PRINT_PREDICTIONS_VALUES.keys()) + '. For additional options refer to '
                            + 'the documentation.')
        parser.add_argument(self.PARAM_STORE_PREDICTIONS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether predictions should be written into output files or not. Must be one of '
                            + format_set(self.STORE_PREDICTIONS_VALUES.keys()) + '. Does only have an effect, if the '
                            + 'parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to '
                            + 'the documentation.')
        parser.add_argument(self.PARAM_PRINT_GROUND_TRUTH,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the ground truth should be printed on the console or not. Must be one of '
                            + format_set(self.PRINT_GROUND_TRUTH_VALUES.keys()) + '. For additional options refer '
                            + 'to the documentation.')
        parser.add_argument(self.PARAM_STORE_GROUND_TRUTH,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the ground truth should be written into output files or not. Must be one of '
                            + format_set(self.STORE_GROUND_TRUTH_VALUES.keys()) + '. Does only have an effect, if '
                            + 'the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional options '
                            + 'refer to the documentation.')
        parser.add_argument(self.PARAM_PREDICTION_TYPE,
                            type=str,
                            default=PredictionType.BINARY.value,
                            help='The type of predictions that should be obtained from the learner. Must be one of '
                            + format_enum_values(PredictionType) + '.')

    def create_experiment(self, args: Namespace):
        """
        See :func:`mlrl.testbed.runnables.Runnable.create_experiment`
        """
        dataset_splitter = self.__create_dataset_splitter(args)
        experiment = self._create_experiment(args, dataset_splitter)
        experiment.add_listeners(*filter(lambda listener: listener is not None, [
            self.__create_clear_output_directory_listener(args, dataset_splitter),
        ]))
        experiment.add_input_readers(*filter(lambda listener: listener is not None, [
            self._create_model_reader(args),
            self._create_parameter_reader(args),
        ]))
        experiment.add_pre_training_output_writers(*filter(lambda listener: listener is not None, [
            self._create_data_characteristics_writer(args),
            self._create_parameter_writer(args),
        ]))
        experiment.add_post_training_output_writers(*filter(lambda listener: listener is not None, [
            self._create_model_writer(args),
            self._create_label_vector_writer(args),
        ]))
        prediction_output_writers = self._create_prediction_output_writers(args, experiment.problem_domain)
        experiment.add_prediction_output_writers(*prediction_output_writers)
        return experiment

    def _create_experiment(self, args, dataset_splitter: DatasetSplitter) -> Experiment:
        """
        May be overridden by subclasses in order to create the `Experiment` that should be run.

        :param args:                The command line arguments
        :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
        :return:                    The `Experiment` that has been created
        """
        problem_domain = self._create_problem_domain(args)
        return SkLearnExperiment(problem_domain=problem_domain, dataset_splitter=dataset_splitter)

    def _create_prediction_output_writers(self, args, problem_domain: ProblemDomain) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the output writers that should be invoked each time
        predictions have been obtained from a model.

        :param args:            The command line arguments
        :param problem_domain:  The problem domain, the experiment is concerned with
        :return:                A list that contains the output writers that have been created
        """
        output_writers = []
        output_writer = self._create_evaluation_writer(args, problem_domain)

        if output_writer:
            output_writers.append(output_writer)

        output_writer = self._create_prediction_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        output_writer = self._create_ground_truth_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        output_writer = self._create_prediction_characteristics_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        return output_writers

    def _create_model_reader(self, args) -> Optional[ModelReader]:
        """
        May be overridden by subclasses in order to create the `ModelReader` that should be used for loading models.

        :param args:    The command line arguments
        :return:        The `ModelReader` that has been created
        """
        model_load_dir = args.model_load_dir
        return ModelReader(PickleFileSource(model_load_dir)) if model_load_dir else None

    def _create_model_writer(self, args) -> Optional[ModelWriter]:
        """
        May be overridden by subclasses in order to create the `ModelWriter` that should be used for saving models.

        :param args:    The command line arguments
        :return:        The `ModelWriter` that has been created
        """
        model_save_dir = args.model_save_dir

        if model_save_dir:
            pickle_sink = PickleFileSink(directory=model_save_dir, create_directory=args.create_output_dir)
            return ModelWriter(exit_on_error=args.exit_on_error).add_sinks(pickle_sink)
        return None

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

    def _create_evaluation_writer(self, args, problem_domain: ProblemDomain) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output evaluation
        results.

        :param args:            The command line arguments
        :param problem_domain:  The problem domain, the experiment is concerned with
        :return:                The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_EVALUATION, args.print_evaluation,
                                                 self.PRINT_EVALUATION_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_EVALUATION, args.store_evaluation,
                                                 self.STORE_EVALUATION_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        if sinks:
            if isinstance(problem_domain, RegressionProblem):
                extractor = RegressionEvaluationDataExtractor()
            elif isinstance(problem_domain, ClassificationProblem) and problem_domain.prediction_type in {
                    PredictionType.SCORES, PredictionType.PROBABILITIES
            }:
                extractor = RankingEvaluationDataExtractor()
            else:
                extractor = ClassificationEvaluationDataExtractor()

            return EvaluationWriter(extractor).add_sinks(*sinks)

        return None

    def _create_parameter_reader(self, args) -> Optional[ParameterReader]:
        """
        May be overridden by subclasses in order to create the `ParameterReader` that should be used for loading
        parameter settings.

        :param args:    The command line arguments
        :return:        The `ParameterReader` that has been created
        """
        parameter_load_dir = args.parameter_load_dir
        return ParameterReader(CsvFileSource(parameter_load_dir)) if parameter_load_dir else None

    def _create_parameter_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output parameter
        settings.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []

        if args.print_parameters or args.print_all:
            sinks.append(LogSink())

        if args.parameter_save_dir:
            sinks.append(CsvFileSink(directory=args.parameter_save_dir, create_directory=args.create_output_dir))

        return ParameterWriter().add_sinks(*sinks) if sinks else None

    def _create_prediction_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output predictions.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_PREDICTIONS, args.print_predictions,
                                                 self.PRINT_PREDICTIONS_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTIONS, args.store_predictions,
                                                 self.STORE_PREDICTIONS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            sinks.append(
                ArffFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        return PredictionWriter().add_sinks(*sinks) if sinks else None

    def _create_ground_truth_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output the ground
        truth.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_GROUND_TRUTH, args.print_ground_truth,
                                                 self.PRINT_GROUND_TRUTH_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_GROUND_TRUTH, args.store_ground_truth,
                                                 self.STORE_GROUND_TRUTH_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                ArffFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        return GroundTruthWriter().add_sinks(*sinks) if sinks else None

    def _create_prediction_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output prediction
        characteristics.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_PREDICTION_CHARACTERISTICS,
                                                 args.print_prediction_characteristics,
                                                 self.PRINT_PREDICTION_CHARACTERISTICS_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTION_CHARACTERISTICS,
                                                 args.store_prediction_characteristics,
                                                 self.STORE_PREDICTION_CHARACTERISTICS_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        return PredictionCharacteristicsWriter().add_sinks(*sinks) if sinks else None

    def _create_data_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output data
        characteristics.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_DATA_CHARACTERISTICS, args.print_data_characteristics,
                                                 self.PRINT_DATA_CHARACTERISTICS_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_DATA_CHARACTERISTICS, args.store_data_characteristics,
                                                 self.STORE_DATA_CHARACTERISTICS_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        return DataCharacteristicsWriter().add_sinks(*sinks) if sinks else None

    def _create_label_vector_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output unique label
        vectors contained in the training data.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_LABEL_VECTORS, args.print_label_vectors,
                                                 self.PRINT_LABEL_VECTORS_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_LABEL_VECTORS, args.store_label_vectors,
                                                 self.STORE_LABEL_VECTORS_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        if sinks:
            return LabelVectorWriter(LabelVectorSetExtractor(), exit_on_error=args.exit_on_error).add_sinks(*sinks)
        return None

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


class RuleLearnerRunnable(SkLearnRunnable):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a rule learner.
    """

    PARAM_INCREMENTAL_EVALUATION = '--incremental-evaluation'

    OPTION_MIN_SIZE = 'min_size'

    OPTION_MAX_SIZE = 'max_size'

    OPTION_STEP_SIZE = 'step_size'

    INCREMENTAL_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_MIN_SIZE, OPTION_MAX_SIZE, OPTION_STEP_SIZE},
        BooleanOption.FALSE.value: {}
    }

    PARAM_PRINT_RULES = '--print-rules'

    PRINT_RULES_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            RuleModelAsText.OPTION_PRINT_FEATURE_NAMES, RuleModelAsText.OPTION_PRINT_OUTPUT_NAMES,
            RuleModelAsText.OPTION_PRINT_NOMINAL_VALUES, RuleModelAsText.OPTION_PRINT_BODIES,
            RuleModelAsText.OPTION_PRINT_HEADS, RuleModelAsText.OPTION_DECIMALS_BODY,
            RuleModelAsText.OPTION_DECIMALS_HEAD
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_RULES = '--store-rules'

    STORE_RULES_VALUES = PRINT_RULES_VALUES

    PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--print-marginal-probability-calibration-model'

    PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--store-marginal-probability-calibration-model'

    STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES

    PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL = '--print-joint-probability-calibration-model'

    PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL = '--store-joint-probability-calibration-model'

    STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES

    PARAM_FEATURE_FORMAT = '--feature-format'

    PARAM_SPARSE_FEATURE_VALUE = '--sparse-feature-value'

    def __init__(self, classifier_type: Optional[type], classifier_config_type: Optional[type],
                 classifier_parameters: Optional[Set[Parameter]], regressor_type: Optional[type],
                 regressor_config_type: Optional[type], regressor_parameters: Optional[Set[Parameter]]):
        """
        :param classifier_type:         The type of the rule learner to be used in classification problems or None, if
                                        classification problems are not supported
        :param classifier_config_type:  The type of the configuration to be used in classification problems or None, if
                                        classification problems are not supported
        :param classifier_parameters:   A set that contains the parameters that may be supported by the rule learner to
                                        be used in regression problems or None, if regression problems are not supported
        :param regressor_type:          The type of the rule learner to be used in regression problems or None, if
                                        regression problems are not supported
        :param regressor_config_type:   The type of the configuration to be used in regression problems or None, if
                                        regression problems are not supported
        :param regressor_parameters:    A set that contains the parameters that may be supported by the rule learner to
                                        be used in regression problems or None, if regression problems are not supported
        """
        self.classifier_type = classifier_type
        self.classifier_config_type = classifier_config_type
        self.classifier_parameters = classifier_parameters
        self.regressor_type = regressor_type
        self.regressor_config_type = regressor_config_type
        self.regressor_parameters = regressor_parameters

    def __create_config_type_and_parameters(self, problem_domain: ProblemDomain):
        if isinstance(problem_domain, ClassificationProblem):
            config_type = self.classifier_config_type
            parameters = self.classifier_parameters
        elif isinstance(problem_domain, RegressionProblem):
            config_type = self.regressor_config_type
            parameters = self.regressor_parameters
        else:
            config_type = None
            parameters = None

        if config_type and parameters:
            return config_type, parameters
        raise RuntimeError('The machine learning algorithm does not support ' + problem_domain.problem_name
                           + ' problems')

    @staticmethod
    def __configure_argument_parser(parser: ArgumentParser, config_type: type, parameters: Set[Parameter]):
        """
        Configure an `ArgumentParser` by taking into account a given set of parameters.

        :param parser:      The `ArgumentParser` to be configured
        :param config_type: The type of the configuration that should support the parameters
        :param parameters:  A set that contains the parameters to be taken into account
        """
        for parameter in parameters:
            try:
                parameter.add_to_argument_parser(parser, config_type)
            except ArgumentError:
                # Argument has already been added, that's okay
                pass

    def configure_arguments(self, parser: ArgumentParser):
        super().configure_arguments(parser)
        parser.add_argument(self.PARAM_INCREMENTAL_EVALUATION,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether models should be evaluated repeatedly, using only a subset of the induced '
                            + 'rules with increasing size, or not. Must be one of ' + format_enum_values(BooleanOption)
                            + '. For additional options refer to the documentation.')
        parser.add_argument('--print-model-characteristics',
                            type=BooleanOption.parse,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of models should be printed on the console or not. Must '
                            + 'be one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--store-model-characteristics',
                            type=BooleanOption.parse,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of models should be written into output files or not. '
                            + 'Must be one of ' + format_enum_values(BooleanOption) + '. Does only have an effect if '
                            + 'the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified.')
        parser.add_argument(self.PARAM_PRINT_RULES,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the induced rules should be printed on the console or not. Must be one of '
                            + format_set(self.PRINT_RULES_VALUES.keys()) + '. For additional options refer to the '
                            + 'documentation.')
        parser.add_argument(self.PARAM_STORE_RULES,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the induced rules should be written into a text file or not. Must be one of '
                            + format_set(self.STORE_RULES_VALUES.keys()) + '. Does only have an effect if the '
                            + 'parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to '
                            + 'the documentation.')
        parser.add_argument(self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of marginal probabilities should be printed on '
                            + 'the console or not. Must be one of ' + format_enum_values(BooleanOption) + '. For '
                            + 'additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of marginal probabilities should be written '
                            + 'into an output file or not. Must be one of ' + format_enum_values(BooleanOption) + '. '
                            + 'Does only have an effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of joint probabilities should be printed on '
                            + 'the console or not. Must be one of ' + format_enum_values(BooleanOption) + '. For '
                            + 'additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of joint probabilities should be written into '
                            + 'an output file or not. Must be one of ' + format_enum_values(BooleanOption) + '. Does '
                            + 'only have an effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For '
                            + 'additional options refer to the documentation.')
        parser.add_argument(self.PARAM_FEATURE_FORMAT,
                            type=str,
                            default=None,
                            help='The format to be used for the representation of the feature matrix. Must be one of '
                            + format_enum_values(SparsePolicy) + '.')
        parser.add_argument(self.PARAM_SPARSE_FEATURE_VALUE,
                            type=float,
                            default=0.0,
                            help='The value that should be used for sparse elements in the feature matrix. Does only '
                            + 'have an effect if a sparse format is used for the representation of the feature matrix, '
                            + 'depending on the parameter ' + self.PARAM_FEATURE_FORMAT + '.')
        parser.add_argument('--output-format',
                            type=str,
                            default=None,
                            help='The format to be used for the representation of the output matrix. Must be one of '
                            + format_enum_values(SparsePolicy) + '.')
        parser.add_argument('--prediction-format',
                            type=str,
                            default=None,
                            help='The format to be used for the representation of predictions. Must be one of '
                            + format_enum_values(SparsePolicy) + '.')
        problem_domain = self._create_problem_domain(parser.parse_known_args()[0])
        config_type, parameters = self.__create_config_type_and_parameters(problem_domain)
        self.__configure_argument_parser(parser, config_type, parameters)

    def _create_experiment(self, args, dataset_splitter: DatasetSplitter) -> Experiment:
        kwargs = {RuleLearner.KWARG_SPARSE_FEATURE_VALUE: args.sparse_feature_value}
        problem_domain = self._create_problem_domain(args, fit_kwargs=kwargs, predict_kwargs=kwargs)
        experiment = SkLearnExperiment(problem_domain=problem_domain, dataset_splitter=dataset_splitter)
        experiment.add_post_training_output_writers(*filter(lambda listener: listener is not None, [
            self._create_model_as_text_writer(args),
            self._create_model_characteristics_writer(args),
            self._create_marginal_probability_calibration_model_writer(args),
            self._create_joint_probability_calibration_model_writer(args),
        ]))
        return experiment

    def create_classifier(self, args) -> Optional[SkLearnClassifierMixin]:
        classifier_type = self.classifier_type

        if classifier_type:
            kwargs = self.__create_kwargs_from_parameters(self.classifier_parameters, args)
            return classifier_type(**kwargs)
        return None

    def create_regressor(self, args) -> Optional[SkLearnRegressorMixin]:
        regressor_type = self.regressor_type

        if regressor_type:
            kwargs = self.__create_kwargs_from_parameters(self.regressor_parameters, args)
            return regressor_type(**kwargs)
        return None

    @staticmethod
    def __create_kwargs_from_parameters(parameters: Set[Parameter], args):
        kwargs = {}
        args_dict = vars(args)

        for parameter in parameters:
            parameter_name = parameter.name

            if parameter_name in args_dict:
                kwargs[parameter_name] = args_dict[parameter_name]

        kwargs['feature_format'] = args.feature_format
        kwargs['output_format'] = args.output_format
        kwargs['prediction_format'] = args.prediction_format
        return kwargs

    def _create_model_as_text_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output textual
        representations of models.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_RULES, args.print_rules, self.PRINT_RULES_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_RULES, args.store_rules, self.STORE_RULES_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                TextFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        if sinks:
            return ModelAsTextWriter(RuleModelAsTextExtractor(), exit_on_error=args.exit_on_error).add_sinks(*sinks)
        return None

    def _create_model_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output the
        characteristics of models.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []

        if args.print_model_characteristics or args.print_all:
            sinks.append(LogSink())

        if (args.store_model_characteristics or args.store_all) and args.output_dir:
            sinks.append(CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir))

        if sinks:
            return ModelCharacteristicsWriter(RuleModelCharacteristicsExtractor(),
                                              exit_on_error=args.exit_on_error).add_sinks(*sinks)
        return None

    def _create_marginal_probability_calibration_model_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output textual
        representations of models for the calibration of marginal probabilities.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.print_marginal_probability_calibration_model,
                                                 self.PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_marginal_probability_calibration_model,
                                                 self.STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        if sinks:
            return ProbabilityCalibrationModelWriter(IsotonicMarginalProbabilityCalibrationModelExtractor(),
                                                     exit_on_error=args.exit_on_error).add_sinks(*sinks)
        return None

    def _create_joint_probability_calibration_model_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output textual
        representations of models for the calibration of joint probabilities.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.print_joint_probability_calibration_model,
                                                 self.PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_joint_probability_calibration_model,
                                                 self.STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        if sinks:
            return ProbabilityCalibrationModelWriter(IsotonicJointProbabilityCalibrationModelExtractor(),
                                                     exit_on_error=args.exit_on_error).add_sinks(*sinks)
        return None

    def _create_predictor_factory(self, args, prediction_type: PredictionType) -> SkLearnProblem.PredictorFactory:
        value, options = parse_param_and_options(self.PARAM_INCREMENTAL_EVALUATION, args.incremental_evaluation,
                                                 self.INCREMENTAL_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value:
            min_size = options.get_int(self.OPTION_MIN_SIZE, 0)
            assert_greater_or_equal(self.OPTION_MIN_SIZE, min_size, 0)
            max_size = options.get_int(self.OPTION_MAX_SIZE, 0)
            if max_size != 0:
                assert_greater(self.OPTION_MAX_SIZE, max_size, min_size)
            step_size = options.get_int(self.OPTION_STEP_SIZE, 1)
            assert_greater_or_equal(self.OPTION_STEP_SIZE, step_size, 1)

            def predictor_factory():
                return IncrementalPredictor(prediction_type, min_size=min_size, max_size=max_size, step_size=step_size)

            return predictor_factory

        return super()._create_predictor_factory(args, prediction_type)
