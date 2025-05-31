"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write evaluation results to one or several sinks.
"""
from argparse import Namespace
from typing import Dict, List, Set

from mlrl.testbed.cli import Argument, BoolArgument
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.evaluation.evaluation_result import EvaluationResult
from mlrl.testbed.experiments.output.evaluation.extractor_classification import ClassificationEvaluationDataExtractor
from mlrl.testbed.experiments.output.evaluation.extractor_ranking import RankingEvaluationDataExtractor
from mlrl.testbed.experiments.output.evaluation.extractor_regression import RegressionEvaluationDataExtractor
from mlrl.testbed.experiments.output.evaluation.writer import EvaluationWriter
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, RegressionProblem
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.options import BooleanOption, parse_param_and_options


class EvaluationExtension(Extension):
    """
    An extension that configures the functionality to write evaluation results to one or several sinks.
    """

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

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            BoolArgument(
                self.PARAM_PRINT_EVALUATION,
                default=True,
                help='Whether the evaluation results should be printed on the console or not.',
                true_options={
                    EvaluationResult.OPTION_ENABLE_ALL, EvaluationResult.OPTION_HAMMING_LOSS,
                    EvaluationResult.OPTION_HAMMING_ACCURACY, EvaluationResult.OPTION_SUBSET_ZERO_ONE_LOSS,
                    EvaluationResult.OPTION_SUBSET_ACCURACY, EvaluationResult.OPTION_MICRO_PRECISION,
                    EvaluationResult.OPTION_MICRO_RECALL, EvaluationResult.OPTION_MICRO_F1,
                    EvaluationResult.OPTION_MICRO_JACCARD, EvaluationResult.OPTION_MACRO_PRECISION,
                    EvaluationResult.OPTION_MACRO_RECALL, EvaluationResult.OPTION_MACRO_F1,
                    EvaluationResult.OPTION_MACRO_JACCARD, EvaluationResult.OPTION_EXAMPLE_WISE_PRECISION,
                    EvaluationResult.OPTION_EXAMPLE_WISE_RECALL, EvaluationResult.OPTION_EXAMPLE_WISE_F1,
                    EvaluationResult.OPTION_EXAMPLE_WISE_JACCARD, EvaluationResult.OPTION_ACCURACY,
                    EvaluationResult.OPTION_ZERO_ONE_LOSS, EvaluationResult.OPTION_PRECISION,
                    EvaluationResult.OPTION_RECALL, EvaluationResult.OPTION_F1, EvaluationResult.OPTION_JACCARD,
                    EvaluationResult.OPTION_MEAN_ABSOLUTE_ERROR, EvaluationResult.OPTION_MEAN_SQUARED_ERROR,
                    EvaluationResult.OPTION_MEDIAN_ABSOLUTE_ERROR,
                    EvaluationResult.OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, EvaluationResult.OPTION_RANK_LOSS,
                    EvaluationResult.OPTION_COVERAGE_ERROR, EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
                    EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
                    EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN,
                    EvaluationResult.OPTION_TRAINING_TIME, EvaluationResult.OPTION_PREDICTION_TIME, OPTION_DECIMALS,
                    OPTION_PERCENTAGE
                },
            ),
            BoolArgument(
                self.PARAM_STORE_EVALUATION,
                default=True,
                help='Whether the evaluation results should be written into output files or not. Does only have an '
                + 'effect if the argument ' + OutputExtension.PARAM_OUTPUT_DIR + ' is specified.',
                true_options={
                    EvaluationResult.OPTION_ENABLE_ALL, EvaluationResult.OPTION_HAMMING_LOSS,
                    EvaluationResult.OPTION_HAMMING_ACCURACY, EvaluationResult.OPTION_SUBSET_ZERO_ONE_LOSS,
                    EvaluationResult.OPTION_SUBSET_ACCURACY, EvaluationResult.OPTION_MICRO_PRECISION,
                    EvaluationResult.OPTION_MICRO_RECALL, EvaluationResult.OPTION_MICRO_F1,
                    EvaluationResult.OPTION_MICRO_JACCARD, EvaluationResult.OPTION_MACRO_PRECISION,
                    EvaluationResult.OPTION_MACRO_RECALL, EvaluationResult.OPTION_MACRO_F1,
                    EvaluationResult.OPTION_MACRO_JACCARD, EvaluationResult.OPTION_EXAMPLE_WISE_PRECISION,
                    EvaluationResult.OPTION_EXAMPLE_WISE_RECALL, EvaluationResult.OPTION_EXAMPLE_WISE_F1,
                    EvaluationResult.OPTION_EXAMPLE_WISE_JACCARD, EvaluationResult.OPTION_ACCURACY,
                    EvaluationResult.OPTION_ZERO_ONE_LOSS, EvaluationResult.OPTION_PRECISION,
                    EvaluationResult.OPTION_RECALL, EvaluationResult.OPTION_F1, EvaluationResult.OPTION_JACCARD,
                    EvaluationResult.OPTION_MEAN_ABSOLUTE_ERROR, EvaluationResult.OPTION_MEAN_SQUARED_ERROR,
                    EvaluationResult.OPTION_MEDIAN_ABSOLUTE_ERROR,
                    EvaluationResult.OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, EvaluationResult.OPTION_RANK_LOSS,
                    EvaluationResult.OPTION_COVERAGE_ERROR, EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
                    EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
                    EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN,
                    EvaluationResult.OPTION_TRAINING_TIME, EvaluationResult.OPTION_PREDICTION_TIME, OPTION_DECIMALS,
                    OPTION_PERCENTAGE
                },
            ),
        ]

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_PRINT_EVALUATION, args.print_evaluation,
                                                 self.PRINT_EVALUATION_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            return [LogSink(options)]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_STORE_EVALUATION, args.store_evaluation,
                                                 self.STORE_EVALUATION_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            return [CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            problem_domain = experiment_builder.problem_domain

            if isinstance(problem_domain, RegressionProblem):
                extractor = RegressionEvaluationDataExtractor()
            elif isinstance(problem_domain, ClassificationProblem) and problem_domain.prediction_type in {
                    PredictionType.SCORES, PredictionType.PROBABILITIES
            }:
                extractor = RankingEvaluationDataExtractor()
            else:
                extractor = ClassificationEvaluationDataExtractor()

            writer = EvaluationWriter(extractor).add_sinks(*sinks)
            experiment_builder.add_prediction_output_writers(writer)
