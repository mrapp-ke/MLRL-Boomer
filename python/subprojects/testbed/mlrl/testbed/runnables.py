"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""
import logging as log
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Optional, Dict, Set

from mlrl.common.config import NONE
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal
from mlrl.common.options import BooleanOption, parse_param_and_options
from mlrl.common.strings import format_enum_values
from mlrl.testbed.data_characteristics import DataCharacteristicsPrinter, DataCharacteristicsLogOutput, \
    DataCharacteristicsCsvOutput
from mlrl.testbed.data_splitting import DataSplitter, CrossValidationSplitter, TrainTestSplitter, NoSplitter, DataSet
from mlrl.testbed.evaluation import ARGUMENT_HAMMING_LOSS, ARGUMENT_HAMMING_ACCURACY, ARGUMENT_SUBSET_ZERO_ONE_LOSS, \
    ARGUMENT_SUBSET_ACCURACY, ARGUMENT_MICRO_PRECISION, ARGUMENT_MICRO_RECALL, ARGUMENT_MICRO_F1, \
    ARGUMENT_MICRO_JACCARD, ARGUMENT_MACRO_PRECISION, ARGUMENT_MACRO_RECALL, ARGUMENT_MACRO_F1, \
    ARGUMENT_MACRO_JACCARD, ARGUMENT_EXAMPLE_WISE_PRECISION, ARGUMENT_EXAMPLE_WISE_RECALL, ARGUMENT_EXAMPLE_WISE_F1, \
    ARGUMENT_EXAMPLE_WISE_JACCARD, ARGUMENT_ACCURACY, ARGUMENT_ZERO_ONE_LOSS, ARGUMENT_PRECISION, ARGUMENT_RECALL, \
    ARGUMENT_F1, ARGUMENT_JACCARD, ARGUMENT_MEAN_ABSOLUTE_ERROR, ARGUMENT_MEAN_SQUARED_ERROR, \
    ARGUMENT_MEDIAN_ABSOLUTE_ERROR, ARGUMENT_MEDIAN_ABSOLUTE_PERCENTAGE_ERROR, ARGUMENT_RANK_LOSS, \
    ARGUMENT_COVERAGE_ERROR, ARGUMENT_LABEL_RANKING_AVERAGE_PRECISION, ARGUMENT_DISCOUNTED_CUMULATIVE_GAIN, \
    ARGUMENT_TRAINING_TIME, ARGUMENT_PREDICTION_TIME, ARGUMENT_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, \
    EvaluationPrinter, ClassificationEvaluationPrinter, ScoreEvaluationPrinter, ProbabilityEvaluationPrinter, \
    EvaluationLogOutput, EvaluationCsvOutput
from mlrl.testbed.experiments import Experiment, PredictionType, Evaluation, GlobalEvaluation, IncrementalEvaluation
from mlrl.testbed.io import clear_directory
from mlrl.testbed.model_characteristics import ARGUMENT_PRINT_FEATURE_NAMES, ARGUMENT_PRINT_LABEL_NAMES, \
    ARGUMENT_PRINT_NOMINAL_VALUES, ARGUMENT_PRINT_BODIES, ARGUMENT_PRINT_HEADS, ModelPrinter, RulePrinter, \
    ModelPrinterLogOutput, ModelPrinterTxtOutput, ModelCharacteristicsPrinter, RuleModelCharacteristicsPrinter, \
    RuleModelCharacteristicsLogOutput, RuleModelCharacteristicsCsvOutput
from mlrl.testbed.parameters import ParameterInput, ParameterCsvInput, ParameterPrinter, ParameterLogOutput, \
    ParameterCsvOutput
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.prediction_characteristics import PredictionCharacteristicsPrinter, \
    PredictionCharacteristicsLogOutput, PredictionCharacteristicsCsvOutput
from mlrl.testbed.predictions import PredictionPrinter, PredictionLogOutput, PredictionArffOutput

LOG_FORMAT = '%(levelname)s %(message)s'

PARAM_DATA_SPLIT = '--data-split'

PARAM_PREDICTION_TYPE = '--prediction-type'

PARAM_PRINT_EVALUATION = '--print-evaluation'

PARAM_STORE_EVALUATION = '--store-evaluation'

PARAM_PRINT_RULES = '--print-rules'

PARAM_STORE_RULES = '--store-rules'

PARAM_INCREMENTAL_EVALUATION = '--incremental-evaluation'

DATA_SPLIT_TRAIN_TEST = 'train-test'

ARGUMENT_TEST_SIZE = 'test_size'

DATA_SPLIT_CROSS_VALIDATION = 'cross-validation'

ARGUMENT_NUM_FOLDS = 'num_folds'

ARGUMENT_CURRENT_FOLD = 'current_fold'

ARGUMENT_MIN_SIZE = 'min_size'

ARGUMENT_MAX_SIZE = 'max_size'

ARGUMENT_STEP_SIZE = 'step_size'

DATA_SPLIT_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    DATA_SPLIT_TRAIN_TEST: {ARGUMENT_TEST_SIZE},
    DATA_SPLIT_CROSS_VALIDATION: {ARGUMENT_NUM_FOLDS, ARGUMENT_CURRENT_FOLD}
}

PRINT_EVALUATION_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_HAMMING_LOSS, ARGUMENT_HAMMING_ACCURACY, ARGUMENT_SUBSET_ZERO_ONE_LOSS,
                               ARGUMENT_SUBSET_ACCURACY, ARGUMENT_MICRO_PRECISION, ARGUMENT_MICRO_RECALL,
                               ARGUMENT_MICRO_F1, ARGUMENT_MICRO_JACCARD, ARGUMENT_MACRO_PRECISION,
                               ARGUMENT_MACRO_RECALL, ARGUMENT_MACRO_F1, ARGUMENT_MACRO_JACCARD,
                               ARGUMENT_EXAMPLE_WISE_PRECISION, ARGUMENT_EXAMPLE_WISE_RECALL, ARGUMENT_EXAMPLE_WISE_F1,
                               ARGUMENT_EXAMPLE_WISE_JACCARD, ARGUMENT_ACCURACY, ARGUMENT_ZERO_ONE_LOSS,
                               ARGUMENT_PRECISION, ARGUMENT_RECALL, ARGUMENT_F1, ARGUMENT_JACCARD,
                               ARGUMENT_MEAN_ABSOLUTE_ERROR, ARGUMENT_MEAN_SQUARED_ERROR,
                               ARGUMENT_MEDIAN_ABSOLUTE_ERROR, ARGUMENT_MEDIAN_ABSOLUTE_PERCENTAGE_ERROR,
                               ARGUMENT_RANK_LOSS, ARGUMENT_COVERAGE_ERROR, ARGUMENT_LABEL_RANKING_AVERAGE_PRECISION,
                               ARGUMENT_DISCOUNTED_CUMULATIVE_GAIN, ARGUMENT_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN},
    BooleanOption.FALSE.value: {}
}

STORE_EVALUATION_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_HAMMING_LOSS, ARGUMENT_HAMMING_ACCURACY, ARGUMENT_SUBSET_ZERO_ONE_LOSS,
                               ARGUMENT_SUBSET_ACCURACY, ARGUMENT_MICRO_PRECISION, ARGUMENT_MICRO_RECALL,
                               ARGUMENT_MICRO_F1, ARGUMENT_MICRO_JACCARD, ARGUMENT_MACRO_PRECISION,
                               ARGUMENT_MACRO_RECALL, ARGUMENT_MACRO_F1, ARGUMENT_MACRO_JACCARD,
                               ARGUMENT_EXAMPLE_WISE_PRECISION, ARGUMENT_EXAMPLE_WISE_RECALL, ARGUMENT_EXAMPLE_WISE_F1,
                               ARGUMENT_EXAMPLE_WISE_JACCARD, ARGUMENT_ACCURACY, ARGUMENT_ZERO_ONE_LOSS,
                               ARGUMENT_PRECISION, ARGUMENT_RECALL, ARGUMENT_F1, ARGUMENT_JACCARD,
                               ARGUMENT_MEAN_ABSOLUTE_ERROR, ARGUMENT_MEAN_SQUARED_ERROR,
                               ARGUMENT_MEDIAN_ABSOLUTE_ERROR, ARGUMENT_MEDIAN_ABSOLUTE_PERCENTAGE_ERROR,
                               ARGUMENT_RANK_LOSS, ARGUMENT_COVERAGE_ERROR, ARGUMENT_LABEL_RANKING_AVERAGE_PRECISION,
                               ARGUMENT_DISCOUNTED_CUMULATIVE_GAIN, ARGUMENT_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN,
                               ARGUMENT_TRAINING_TIME, ARGUMENT_PREDICTION_TIME},
    BooleanOption.FALSE.value: {}
}

PRINT_RULES_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_PRINT_FEATURE_NAMES, ARGUMENT_PRINT_LABEL_NAMES, ARGUMENT_PRINT_NOMINAL_VALUES,
                               ARGUMENT_PRINT_BODIES, ARGUMENT_PRINT_HEADS},
    BooleanOption.FALSE.value: {}
}

STORE_RULES_VALUES = PRINT_RULES_VALUES

INCREMENTAL_EVALUATION_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_MIN_SIZE, ARGUMENT_MAX_SIZE, ARGUMENT_STEP_SIZE},
    BooleanOption.FALSE.value: {}
}


class Runnable(ABC):
    """
    A base class for all programs that can be configured via command line arguments.
    """

    def run(self, parser: ArgumentParser):
        args = parser.parse_args()

        # Configure the logger...
        log_level = args.log_level
        root = log.getLogger()
        root.setLevel(log_level)
        out_handler = log.StreamHandler(sys.stdout)
        out_handler.setLevel(log_level)
        out_handler.setFormatter(log.Formatter(LOG_FORMAT))
        root.addHandler(out_handler)

        self._run(args)

    @abstractmethod
    def _run(self, args):
        """
        Must be implemented by subclasses in order to run the program.

        :param args: The command line arguments
        """
        pass


class LearnerRunnable(Runnable, ABC):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a learner.
    """

    class ClearOutputDirHook(Experiment.ExecutionHook):
        """
        Deletes all files from the output directory before an experiment starts.
        """

        def __init__(self, output_dir: str):
            self.output_dir = output_dir

        def execute(self):
            clear_directory(self.output_dir)

    @staticmethod
    def __create_prediction_type(args) -> PredictionType:
        prediction_type = args.prediction_type

        try:
            return PredictionType(prediction_type)
        except ValueError:
            raise ValueError('Invalid value given for parameter "' + PARAM_PREDICTION_TYPE + '": Must be one of '
                             + format_enum_values(PredictionType) + ', but is "' + str(prediction_type) + '"')

    @staticmethod
    def __create_data_splitter(args) -> DataSplitter:
        data_set = DataSet(data_dir=args.data_dir, data_set_name=args.dataset,
                           use_one_hot_encoding=args.one_hot_encoding)
        value, options = parse_param_and_options(PARAM_DATA_SPLIT, args.data_split, DATA_SPLIT_VALUES)

        if value == DATA_SPLIT_CROSS_VALIDATION:
            num_folds = options.get_int(ARGUMENT_NUM_FOLDS, 10)
            assert_greater_or_equal(ARGUMENT_NUM_FOLDS, num_folds, 2)
            current_fold = options.get_int(ARGUMENT_CURRENT_FOLD, 0)
            if current_fold != 0:
                assert_greater_or_equal(ARGUMENT_CURRENT_FOLD, current_fold, 1)
                assert_less_or_equal(ARGUMENT_CURRENT_FOLD, current_fold, num_folds)
            return CrossValidationSplitter(data_set, num_folds=num_folds, current_fold=current_fold - 1,
                                           random_state=args.random_state)
        elif value == DATA_SPLIT_TRAIN_TEST:
            test_size = options.get_float(ARGUMENT_TEST_SIZE, 0.33)
            assert_greater(ARGUMENT_TEST_SIZE, test_size, 0)
            assert_less(ARGUMENT_TEST_SIZE, test_size, 1)
            return TrainTestSplitter(data_set, test_size=test_size, random_state=args.random_state)
        else:
            return NoSplitter(data_set)

    @staticmethod
    def __create_pre_execution_hook(args, data_splitter: DataSplitter) -> Optional[Experiment.ExecutionHook]:
        current_fold = data_splitter.current_fold if isinstance(data_splitter, CrossValidationSplitter) else -1
        return None if args.output_dir is None or current_fold >= 0 else LearnerRunnable.ClearOutputDirHook(
            output_dir=args.output_dir)

    @staticmethod
    def __create_parameter_input(args) -> Optional[ParameterInput]:
        return None if args.parameter_dir is None else ParameterCsvInput(input_dir=args.parameter_dir)

    @staticmethod
    def __create_parameter_printer(args) -> Optional[ParameterPrinter]:
        outputs = []

        if args.print_parameters:
            outputs.append(ParameterLogOutput())

        if args.store_parameters and args.output_dir is not None:
            outputs.append(ParameterCsvOutput(output_dir=args.output_dir))

        return ParameterPrinter(outputs) if len(outputs) > 0 else None

    @staticmethod
    def __create_persistence(args) -> Optional[ModelPersistence]:
        return None if args.model_dir is None else ModelPersistence(model_dir=args.model_dir)

    @staticmethod
    def __create_evaluation_printer(args, prediction_type: PredictionType) -> Optional[EvaluationPrinter]:
        outputs = []
        value, options = parse_param_and_options(PARAM_PRINT_EVALUATION, args.print_evaluation, PRINT_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value:
            outputs.append(EvaluationLogOutput(options))

        value, options = parse_param_and_options(PARAM_STORE_EVALUATION, args.store_evaluation, STORE_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            outputs.append(EvaluationCsvOutput(options, output_dir=args.output_dir))

        if len(outputs) > 0:
            if prediction_type == PredictionType.SCORES:
                evaluation = ScoreEvaluationPrinter(outputs)
            elif prediction_type == PredictionType.PROBABILITIES:
                evaluation = ProbabilityEvaluationPrinter(outputs)
            else:
                evaluation = ClassificationEvaluationPrinter(outputs)
        else:
            evaluation = None

        return evaluation

    @staticmethod
    def __create_prediction_printer(args) -> Optional[PredictionPrinter]:
        outputs = []

        if args.print_predictions:
            outputs.append(PredictionLogOutput())

        if args.store_predictions and args.output_dir is not None:
            outputs.append(PredictionArffOutput(output_dir=args.output_dir))

        return PredictionPrinter(outputs) if len(outputs) > 0 else None

    @staticmethod
    def __create_prediction_characteristics_printer(args) -> Optional[PredictionCharacteristicsPrinter]:
        outputs = []

        if args.print_prediction_characteristics:
            outputs.append(PredictionCharacteristicsLogOutput())

        if args.store_prediction_characteristics and args.output_dir is not None:
            outputs.append(PredictionCharacteristicsCsvOutput(output_dir=args.output_dir))

        return PredictionCharacteristicsPrinter(outputs=outputs) if len(outputs) > 0 else None

    @staticmethod
    def __create_data_characteristics_printer(args) -> (Optional[DataCharacteristicsPrinter], bool):
        outputs = []

        if args.print_data_characteristics:
            outputs.append(DataCharacteristicsLogOutput())

        if args.store_data_characteristics and args.output_dir is not None:
            outputs.append(DataCharacteristicsCsvOutput(output_dir=args.output_dir))

        return DataCharacteristicsPrinter(outputs=outputs) if len(outputs) > 0 else None

    def _run(self, args):
        prediction_type = self.__create_prediction_type(args)

        if args.evaluate_training_data:
            train_evaluation_printer = self.__create_evaluation_printer(args, prediction_type)
            train_prediction_printer = self.__create_prediction_printer(args)
            train_prediction_characteristics_printer = self.__create_prediction_characteristics_printer(args)
        else:
            train_evaluation_printer = None
            train_prediction_printer = None
            train_prediction_characteristics_printer = None

        train_evaluation = self._create_evaluation(args, prediction_type, train_evaluation_printer,
                                                   train_prediction_printer, train_prediction_characteristics_printer)
        test_evaluation = self._create_evaluation(args, prediction_type,
                                                  self.__create_evaluation_printer(args, prediction_type),
                                                  self.__create_prediction_printer(args),
                                                  self.__create_prediction_characteristics_printer(args))
        data_splitter = self.__create_data_splitter(args)
        experiment = Experiment(base_learner=self._create_learner(args),
                                learner_name=self._get_learner_name(),
                                data_splitter=data_splitter,
                                pre_execution_hook=self.__create_pre_execution_hook(args, data_splitter),
                                train_evaluation=train_evaluation,
                                test_evaluation=test_evaluation,
                                parameter_input=self.__create_parameter_input(args),
                                parameter_printer=self.__create_parameter_printer(args),
                                model_printer=self._create_model_printer(args),
                                model_characteristics_printer=self._create_model_characteristics_printer(args),
                                data_characteristics_printer=self.__create_data_characteristics_printer(args),
                                persistence=self.__create_persistence(args))
        experiment.run()

    def _create_evaluation(
            self, args, prediction_type: PredictionType, evaluation_printer: Optional[EvaluationPrinter],
            prediction_printer: Optional[PredictionPrinter],
            prediction_characteristics_printer: Optional[PredictionCharacteristicsPrinter]) -> Optional[Evaluation]:
        """
        May be overridden by subclasses in order to create the `Evaluation` that should be used to evaluate predictions
        that are obtained from a previously trained model.

        :param args:                                The command line arguments
        :param prediction_type:                     The type of the predictions to be obtained
        :param evaluation_printer:                  The printer to be used for evaluating the predictions or None, if
                                                    the predictions should not be evaluated
        :param prediction_printer:                  The printer to be used for printing the predictions or None, if the
                                                    predictions should not be printed
        :param prediction_characteristics_printer:  The printer to be used for printing the characteristics of the
                                                    predictions or None, if the characteristics should not be printed
        :return:                                    The `Evaluation` that has been created
        """
        if evaluation_printer is not None or prediction_printer is not None \
                or prediction_characteristics_printer is not None:
            return GlobalEvaluation(prediction_type, evaluation_printer, prediction_printer,
                                    prediction_characteristics_printer)
        else:
            return None

    def _create_model_printer(self, args) -> Optional[ModelPrinter]:
        """
        May be overridden by subclasses in order to create the `ModelPrinter` that should be used to print textual
        representations of models.

        :param args:    The command line arguments
        :return:        The `ModelPrinter` that has been created
        """
        log.warning('The learner does not support printing textual representations of models')
        return None

    def _create_model_characteristics_printer(self, args) -> Optional[ModelCharacteristicsPrinter]:
        """
        May be overridden by subclasses in order to create the `ModelCharacteristicsPrinter` that should be used to
        print the characteristics of models.

        :param args:    The command line arguments
        :return:        The `ModelCharacteristicsPrinter` that has been created
        """
        log.warning('The learner does not support printing the characteristics of models')
        return None

    @abstractmethod
    def _create_learner(self, args):
        """
        Must be implemented by subclasses in order to create the learner.

        :param args:    The command line arguments
        :return:        The learner that has been created
        """
        pass

    @abstractmethod
    def _get_learner_name(self) -> str:
        """
        Must be implemented by subclasses in order to provide the name of the learner.

        :return: The name of the learner
        """
        pass


class RuleLearnerRunnable(LearnerRunnable, ABC):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a rule learner.
    """

    def _create_evaluation(
            self, args, prediction_type: PredictionType, evaluation_printer: Optional[EvaluationPrinter],
            prediction_printer: Optional[PredictionPrinter],
            prediction_characteristics_printer: Optional[PredictionCharacteristicsPrinter]) -> Optional[Evaluation]:
        value, options = parse_param_and_options(PARAM_INCREMENTAL_EVALUATION, args.incremental_evaluation,
                                                 INCREMENTAL_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value:
            min_size = options.get_int(ARGUMENT_MIN_SIZE, 1)
            assert_greater_or_equal(ARGUMENT_MIN_SIZE, min_size, 1)
            max_size = options.get_int(ARGUMENT_MAX_SIZE, 0)
            if max_size != 0:
                assert_greater(ARGUMENT_MAX_SIZE, max_size, min_size)
            step_size = options.get_int(ARGUMENT_STEP_SIZE, 1)
            assert_greater_or_equal(ARGUMENT_STEP_SIZE, step_size, 1)

            if evaluation_printer is not None or prediction_printer is not None \
                    or prediction_characteristics_printer is not None:
                return IncrementalEvaluation(prediction_type, evaluation_printer, prediction_printer,
                                             prediction_characteristics_printer)
            else:
                return None
        else:
            return super()._create_evaluation(args, prediction_type, evaluation_printer, prediction_printer,
                                              prediction_characteristics_printer)

    def _create_model_printer(self, args) -> Optional[ModelPrinter]:
        outputs = []
        value, options = parse_param_and_options(PARAM_PRINT_RULES, args.print_rules, PRINT_RULES_VALUES)

        if value == BooleanOption.TRUE.value:
            outputs.append(ModelPrinterLogOutput(options))

        value, options = parse_param_and_options(PARAM_STORE_RULES, args.store_rules, STORE_RULES_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            outputs.append(ModelPrinterTxtOutput(options, output_dir=args.output_dir))

        return RulePrinter(outputs) if len(outputs) > 0 else None

    def _create_model_characteristics_printer(self, args) -> Optional[ModelCharacteristicsPrinter]:
        outputs = []

        if args.print_model_characteristics:
            outputs.append(RuleModelCharacteristicsLogOutput())

        if args.store_model_characteristics and args.output_dir is not None:
            outputs.append(RuleModelCharacteristicsCsvOutput(output_dir=args.output_dir))

        return RuleModelCharacteristicsPrinter(outputs) if len(outputs) > 0 else None
