"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write evaluation results to one or several sinks.
"""
from argparse import Namespace
from typing import Set

from mlrl.testbed_sklearn.experiments.output.evaluation.evaluation_result import EvaluationResult
from mlrl.testbed_sklearn.experiments.output.evaluation.extractor_classification import \
    ClassificationEvaluationDataExtractor
from mlrl.testbed_sklearn.experiments.output.evaluation.extractor_ranking import RankingEvaluationDataExtractor
from mlrl.testbed_sklearn.experiments.output.evaluation.extractor_regression import RegressionEvaluationDataExtractor

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, RegressionProblem
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.cli import Argument, BoolArgument


class EvaluationExtension(Extension):
    """
    An extension that configures the functionality to write evaluation results to one or several sinks.
    """

    PRINT_EVALUATION = BoolArgument(
        '--print-evaluation',
        default=True,
        description='Whether the evaluation results should be printed on the console or not.',
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
            EvaluationResult.OPTION_ZERO_ONE_LOSS, EvaluationResult.OPTION_PRECISION, EvaluationResult.OPTION_RECALL,
            EvaluationResult.OPTION_F1, EvaluationResult.OPTION_JACCARD, EvaluationResult.OPTION_MEAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_SQUARED_ERROR, EvaluationResult.OPTION_MEDIAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, EvaluationResult.OPTION_RANK_LOSS,
            EvaluationResult.OPTION_COVERAGE_ERROR, EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
            EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, EvaluationResult.OPTION_TRAINING_TIME,
            EvaluationResult.OPTION_PREDICTION_TIME, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
    )

    STORE_EVALUATION = BoolArgument(
        '--store-evaluation',
        default=True,
        description='Whether the evaluation results should be written into output files or not. Does only have an '
        + 'effect if the argument ' + OutputExtension.OUTPUT_DIR.name + ' is specified.',
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
            EvaluationResult.OPTION_ZERO_ONE_LOSS, EvaluationResult.OPTION_PRECISION, EvaluationResult.OPTION_RECALL,
            EvaluationResult.OPTION_F1, EvaluationResult.OPTION_JACCARD, EvaluationResult.OPTION_MEAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_SQUARED_ERROR, EvaluationResult.OPTION_MEDIAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, EvaluationResult.OPTION_RANK_LOSS,
            EvaluationResult.OPTION_COVERAGE_ERROR, EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
            EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, EvaluationResult.OPTION_TRAINING_TIME,
            EvaluationResult.OPTION_PREDICTION_TIME, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), *dependencies)

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_EVALUATION, self.STORE_EVALUATION}

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        print_all = OutputExtension.PRINT_ALL.get_value(args)
        print_evaluation, options = self.PRINT_EVALUATION.get_value(args, default=print_all)

        if print_evaluation:
            experiment_builder.evaluation_writer.add_sinks(LogSink(options))

    def __configure_csv_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        store_all = OutputExtension.STORE_ALL.get_value(args)
        store_evaluation, options = self.STORE_EVALUATION.get_value(args, default=store_all)
        output_directory = OutputExtension.OUTPUT_DIR.get_value(args)

        if store_evaluation and output_directory:
            create_output_directory = OutputExtension.CREATE_OUTPUT_DIR.get_value(args)
            experiment_builder.evaluation_writer.add_sinks(
                CsvFileSink(directory=output_directory, create_directory=create_output_directory, options=options))

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_csv_file_sink(args, experiment_builder)

        if experiment_builder.evaluation_writer.sinks:
            problem_domain = experiment_builder.problem_domain

            if isinstance(problem_domain, RegressionProblem):
                extractor = RegressionEvaluationDataExtractor()
            elif isinstance(problem_domain, ClassificationProblem) and problem_domain.prediction_type in {
                    PredictionType.SCORES, PredictionType.PROBABILITIES
            }:
                extractor = RankingEvaluationDataExtractor()
            else:
                extractor = ClassificationEvaluationDataExtractor()

            experiment_builder.evaluation_writer.extractors.append(extractor)
