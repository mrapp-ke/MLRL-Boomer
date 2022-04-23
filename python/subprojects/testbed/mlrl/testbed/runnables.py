"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""
import logging as log
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Optional

from mlrl.testbed.data_characteristics import DataCharacteristicsPrinter, DataCharacteristicsLogOutput, \
    DataCharacteristicsCsvOutput
from mlrl.testbed.evaluation import Evaluation, ClassificationEvaluation, RankingEvaluation, EvaluationLogOutput, \
    EvaluationCsvOutput
from mlrl.testbed.experiments import Experiment
from mlrl.testbed.model_characteristics import ModelPrinter, RulePrinter, ModelPrinterLogOutput, \
    ModelPrinterTxtOutput, ModelCharacteristicsPrinter, RuleModelCharacteristicsPrinter, \
    RuleModelCharacteristicsLogOutput, RuleModelCharacteristicsCsvOutput
from mlrl.testbed.parameters import ParameterInput, ParameterCsvInput
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.prediction_characteristics import PredictionCharacteristicsPrinter, \
    PredictionCharacteristicsLogOutput, PredictionCharacteristicsCsvOutput
from mlrl.testbed.predictions import PredictionPrinter, PredictionLogOutput, PredictionArffOutput
from mlrl.testbed.training import DataSet

LOG_FORMAT = '%(levelname)s %(message)s'


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

        log.info('Configuration: %s', args)
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

    @staticmethod
    def __create_data_set(args) -> DataSet:
        return DataSet(data_dir=args.data_dir, data_set_name=args.dataset, use_one_hot_encoding=args.one_hot_encoding)

    @staticmethod
    def __create_parameter_input(args) -> Optional[ParameterInput]:
        return None if args.parameter_dir is None else ParameterCsvInput(input_dir=args.parameter_dir)

    @staticmethod
    def __create_persistence(args) -> Optional[ModelPersistence]:
        return None if args.model_dir is None else ModelPersistence(model_dir=args.model_dir)

    @staticmethod
    def __create_evaluation(args, clear_output_dir: bool) -> (Optional[Evaluation], bool):
        outputs = []

        if args.print_evaluation:
            outputs.append(EvaluationLogOutput())

        if args.store_evaluation and args.output_dir is not None:
            outputs.append(EvaluationCsvOutput(output_dir=args.output_dir, clear_dir=clear_output_dir))
            clear_output_dir = False

        if len(outputs) > 0:
            if args.predict_probabilities:
                evaluation = RankingEvaluation(outputs)
            else:
                evaluation = ClassificationEvaluation(outputs)
        else:
            evaluation = None

        return evaluation, clear_output_dir

    @staticmethod
    def __create_prediction_printer(args, clear_output_dir: bool) -> (Optional[PredictionPrinter], bool):
        outputs = []

        if args.print_predictions:
            outputs.append(PredictionLogOutput())

        if args.store_predictions and args.output_dir is not None:
            outputs.append(PredictionArffOutput(output_dir=args.output_dir, clear_dir=clear_output_dir))
            clear_output_dir = False

        printer = PredictionPrinter(outputs) if len(outputs) > 0 else None
        return printer, clear_output_dir

    @staticmethod
    def __create_prediction_characteristics_printer(
            args, clear_output_dir: bool) -> (Optional[PredictionCharacteristicsPrinter], bool):
        outputs = []

        if args.print_prediction_characteristics:
            outputs.append(PredictionCharacteristicsLogOutput())

        if args.store_prediction_characteristics and args.output_dir is not None:
            outputs.append(PredictionCharacteristicsCsvOutput(output_dir=args.output_dir, clear_dir=clear_output_dir))
            clear_output_dir = False

        printer = PredictionCharacteristicsPrinter(outputs=outputs) if len(outputs) > 0 else None
        return printer, clear_output_dir

    @staticmethod
    def __create_data_characteristics_printer(args,
                                              clear_output_dir: bool) -> (Optional[DataCharacteristicsPrinter], bool):
        outputs = []

        if args.print_data_characteristics:
            outputs.append(DataCharacteristicsLogOutput())

        if args.store_data_characteristics and args.output_dir is not None:
            outputs.append(DataCharacteristicsCsvOutput(output_dir=args.output_dir, clear_dir=clear_output_dir))
            clear_output_dir = False

        printer = DataCharacteristicsPrinter(outputs=outputs) if len(outputs) > 0 else None
        return printer, clear_output_dir

    def _run(self, args):
        # Create outputs...
        clear_output_dir = args.current_fold < 0
        data_characteristics_printer, clear_output_dir = self.__create_data_characteristics_printer(
            args, clear_output_dir)

        if args.evaluate_training_data:
            train_evaluation, clear_output_dir = self.__create_evaluation(args, clear_output_dir)
            train_prediction_printer, clear_output_dir = self.__create_prediction_printer(args, clear_output_dir)
            train_prediction_characteristics_printer, clear_output_dir = \
                self.__create_prediction_characteristics_printer(args, clear_output_dir)
        else:
            train_evaluation = None
            train_prediction_printer = None
            train_prediction_characteristics_printer = None

        test_evaluation, clear_output_dir = self.__create_evaluation(args, clear_output_dir)
        test_prediction_printer, clear_output_dir = self.__create_prediction_printer(args, clear_output_dir)
        test_prediction_characteristics_printer, clear_output_dir = self.__create_prediction_characteristics_printer(
            args, clear_output_dir)
        model_characteristics_printer, clear_output_dir = self._create_model_characteristics_printer(
            args, clear_output_dir)
        model_printer, clear_output_dir = self._create_model_printer(args, clear_output_dir)


        # Configure experiment...
        experiment = Experiment(base_learner=self._create_learner(args),
                                predict_probabilities=args.predict_probabilities,
                                test_evaluation=test_evaluation,
                                train_evaluation=train_evaluation,
                                train_prediction_printer=train_prediction_printer,
                                test_prediction_printer=test_prediction_printer,
                                train_prediction_characteristics_printer=train_prediction_characteristics_printer,
                                test_prediction_characteristics_printer=test_prediction_characteristics_printer,
                                data_set=self.__create_data_set(args),
                                num_folds=args.folds,
                                current_fold=args.current_fold,
                                parameter_input=self.__create_parameter_input(args),
                                model_printer=model_printer,
                                model_characteristics_printer=model_characteristics_printer,
                                data_characteristics_printer=data_characteristics_printer,
                                persistence=self.__create_persistence(args))
        experiment.random_state = args.random_state
        experiment.run()

    def _create_model_printer(self, args, clear_output_dir: bool) -> (Optional[ModelPrinter], bool):
        """
        May be overridden by subclasses in order to create the `ModelPrinter` that should be used to print textual
        representations of models.

        :param args:                The command line arguments
        :param clear_output_dir:    True, if the output dir should be cleared before writing output files, False
                                    otherwise
        :return:                    The `ModelPrinter` that has been created and whether it clears the output directory
                                    before writing output files
        """
        log.warning('The learner does not support printing textual representations of models')
        return None, False

    def _create_model_characteristics_printer(self, args,
                                              clear_output_dir: bool) -> (Optional[ModelCharacteristicsPrinter], bool):
        """
        May be overridden by subclasses in order to create the `ModelCharacteristicsPrinter` that should be used to
        print the characteristics of models.

        :param args:                The command line arguments
        :param clear_output_dir:    True, if the output directory should be cleared before writing output files, False
                                    otherwise
        :return:                    The `ModelCharacteristicsPrinter` that has been created and whether it clears the
                                    output directory before writing output files
        """
        log.warning('The learner does not support printing the characteristics of models')
        return None, False

    @abstractmethod
    def _create_learner(self, args):
        """
        Must be implemented by subclasses in order to create the learner.

        :param args:    The command line arguments
        :return:        The learner that has been created
        """
        pass


class RuleLearnerRunnable(LearnerRunnable, ABC):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a rule learner.
    """

    def _create_model_printer(self, args, clear_output_dir: bool) -> (Optional[ModelPrinter], bool):
        outputs = []

        if args.print_rules:
            outputs.append(ModelPrinterLogOutput())

        if args.store_rules and args.output_dir is not None:
            outputs.append(ModelPrinterTxtOutput(output_dir=args.output_dir, clear_dir=clear_output_dir))
            clear_output_dir = False

        printer = RulePrinter(args.print_options, outputs) if len(outputs) > 0 else None
        return printer, clear_output_dir

    def _create_model_characteristics_printer(self, args,
                                              clear_output_dir: bool) -> (Optional[ModelCharacteristicsPrinter], bool):
        outputs = []

        if args.print_model_characteristics:
            outputs.append(RuleModelCharacteristicsLogOutput())

        if args.store_model_characteristics and args.output_dir is not None:
            outputs.append(RuleModelCharacteristicsCsvOutput(output_dir=args.output_dir, clear_dir=clear_output_dir))
            clear_output_dir = False

        printer = RuleModelCharacteristicsPrinter(outputs) if len(outputs) > 0 else None
        return printer, clear_output_dir
