"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to write evaluation results to different sinks.
"""
from mlrl.testbed.experiments.output.evaluation.evaluation_result import EvaluationResult
from mlrl.testbed.experiments.output.evaluation.extractor_classification import ClassificationEvaluationDataExtractor
from mlrl.testbed.experiments.output.evaluation.extractor_ranking import RankingEvaluationDataExtractor
from mlrl.testbed.experiments.output.evaluation.extractor_regression import RegressionEvaluationDataExtractor
from mlrl.testbed.experiments.output.evaluation.writer import EvaluationWriter
