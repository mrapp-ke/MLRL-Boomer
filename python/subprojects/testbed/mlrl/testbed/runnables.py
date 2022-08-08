"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""
import logging as log
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Optional, Dict, Set

from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal
from mlrl.common.options import BooleanOption, parse_param_and_options
from mlrl.testbed.data_characteristics import DataCharacteristicsPrinter, DataCharacteristicsLogOutput, \
    DataCharacteristicsCsvOutput
from mlrl.testbed.data_splitting import DataSplitter, CrossValidationSplitter, TrainTestSplitter, DataSet
from mlrl.testbed.evaluation import Evaluation, ClassificationEvaluation, RankingEvaluation, EvaluationLogOutput, \
    EvaluationCsvOutput
from mlrl.testbed.experiments import Experiment
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

DATA_SPLIT_TRAIN_TEST = 'train-test'

ARGUMENT_TEST_SIZE = 'test_size'

DATA_SPLIT_CROSS_VALIDATION = 'cross-validation'

ARGUMENT_NUM_FOLDS = 'num_folds'

ARGUMENT_CURRENT_FOLD = 'current_fold'

DATA_SPLIT_VALUES: Dict[str, Set[str]] = {
    DATA_SPLIT_TRAIN_TEST: {ARGUMENT_TEST_SIZE},
    DATA_SPLIT_CROSS_VALIDATION: {ARGUMENT_NUM_FOLDS, ARGUMENT_CURRENT_FOLD}
}

PRINT_RULES_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_PRINT_FEATURE_NAMES, ARGUMENT_PRINT_LABEL_NAMES, ARGUMENT_PRINT_NOMINAL_VALUES,
                               ARGUMENT_PRINT_BODIES, ARGUMENT_PRINT_HEADS},
    BooleanOption.FALSE.value: {}
}

STORE_RULES_VALUES = PRINT_RULES_VALUES


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
    def __create_data_splitter(args) -> DataSplitter:
        data_set = DataSet(data_dir=args.data_dir, data_set_name=args.dataset,
                           use_one_hot_encoding=args.one_hot_encoding)
        random_state = args.random_state
        value, options = parse_param_and_options('--data-split', args.data_split, DATA_SPLIT_VALUES)

        if value == DATA_SPLIT_CROSS_VALIDATION:
            num_folds = options.get_int(ARGUMENT_NUM_FOLDS, 10)
            assert_greater_or_equal('num_folds', num_folds, 2)
            current_fold = options.get_int(ARGUMENT_CURRENT_FOLD, 0)
            if current_fold != 0:
                assert_greater_or_equal('current_fold', current_fold, 1)
                assert_less_or_equal('current_fold', current_fold, num_folds)
            return CrossValidationSplitter(data_set, num_folds=num_folds, current_fold=current_fold - 1,
                                           random_state=random_state)
        else:
            test_size = options.get_float(ARGUMENT_TEST_SIZE, 0.33)
            assert_greater('test_size', test_size, 0)
            assert_less('test_size', test_size, 1)
            return TrainTestSplitter(data_set, test_size=test_size, random_state=random_state)

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
    def __create_evaluation(args) -> Optional[Evaluation]:
        outputs = []

        if args.print_evaluation:
            outputs.append(EvaluationLogOutput())

        if args.store_evaluation and args.output_dir is not None:
            outputs.append(EvaluationCsvOutput(output_dir=args.output_dir))

        if len(outputs) > 0:
            if args.predict_probabilities:
                evaluation = RankingEvaluation(outputs)
            else:
                evaluation = ClassificationEvaluation(outputs)
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
        # Create outputs...
        if args.evaluate_training_data:
            train_evaluation = self.__create_evaluation(args)
            train_prediction_printer = self.__create_prediction_printer(args)
            train_prediction_characteristics_printer = self.__create_prediction_characteristics_printer(args)
        else:
            train_evaluation = None
            train_prediction_printer = None
            train_prediction_characteristics_printer = None

        test_prediction_characteristics_printer = self.__create_prediction_characteristics_printer(args)
        data_splitter = self.__create_data_splitter(args)

        # Configure experiment...
        experiment = Experiment(base_learner=self._create_learner(args),
                                learner_name=self._get_learner_name(),
                                data_splitter=data_splitter,
                                pre_execution_hook=self.__create_pre_execution_hook(args, data_splitter),
                                predict_probabilities=args.predict_probabilities,
                                test_evaluation=self.__create_evaluation(args),
                                train_evaluation=train_evaluation,
                                train_prediction_printer=train_prediction_printer,
                                test_prediction_printer=self.__create_prediction_printer(args),
                                train_prediction_characteristics_printer=train_prediction_characteristics_printer,
                                test_prediction_characteristics_printer=test_prediction_characteristics_printer,
                                parameter_input=self.__create_parameter_input(args),
                                parameter_printer=self.__create_parameter_printer(args),
                                model_printer=self._create_model_printer(args),
                                model_characteristics_printer=self._create_model_characteristics_printer(args),
                                data_characteristics_printer=self.__create_data_characteristics_printer(args),
                                persistence=self.__create_persistence(args))
        experiment.run()

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

    def _create_model_printer(self, args) -> Optional[ModelPrinter]:
        outputs = []

        value, options = parse_param_and_options('--print-rules', args.print_rules, PRINT_RULES_VALUES)

        if value == BooleanOption.TRUE.value:
            outputs.append(ModelPrinterLogOutput(options))

        value, options = parse_param_and_options('--store-rules', args.store_rules, STORE_RULES_VALUES)

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
