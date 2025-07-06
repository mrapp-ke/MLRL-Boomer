"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from typing import Any, Optional

import pytest

from ..common.cmd_runner import CmdRunner
from ..common.datasets import Dataset
from ..common.integration_tests_classification import ClassificationIntegrationTests
from .cmd_builder_classification import BoomerClassifierCmdBuilder
from .integration_tests import BoomerIntegrationTestsMixin

from mlrl.common.config.parameters import BINNING_EQUAL_WIDTH, SAMPLING_STRATIFIED_EXAMPLE_WISE, \
    SAMPLING_STRATIFIED_OUTPUT_WISE, GlobalPruningParameter
from mlrl.common.learners import SparsePolicy

from mlrl.boosting.config.parameters import BinaryPredictorParameter, ClassificationLossParameter, HeadTypeParameter, \
    ProbabilityPredictorParameter, StatisticTypeParameter

from mlrl.testbed.experiments.prediction_type import PredictionType

from mlrl.util.cli import NONE


@pytest.mark.boosting
@pytest.mark.classification
class TestBoomerClassifier(ClassificationIntegrationTests, BoomerIntegrationTestsMixin):
    """
    Defines a series of integration tests for the BOOMER algorithm for classification problems.
    """

    def _create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        return BoomerClassifierCmdBuilder(dataset=dataset)

    @pytest.mark.parametrize('prediction_type', [
        PredictionType.SCORES,
        PredictionType.PROBABILITIES,
    ])
    def test_single_label(self, prediction_type: PredictionType, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.single_output) \
            .prediction_type(prediction_type) \
            .print_evaluation()
        CmdRunner(builder).run(f'single-label-{prediction_type}')

    @pytest.mark.parametrize('loss', [
        ClassificationLossParameter.LOSS_LOGISTIC_DECOMPOSABLE,
        ClassificationLossParameter.LOSS_LOGISTIC_NON_DECOMPOSABLE,
        ClassificationLossParameter.LOSS_SQUARED_HINGE_DECOMPOSABLE,
        ClassificationLossParameter.LOSS_SQUARED_HINGE_NON_DECOMPOSABLE,
    ])
    @pytest.mark.parametrize('statistic_type', [
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT32,
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT64,
    ])
    def test_loss(self, loss: str, statistic_type: str):
        builder = self._create_cmd_builder() \
            .loss(loss) \
            .statistic_type(statistic_type)
        CmdRunner(builder).run(f'loss-{loss}_{statistic_type}-statistics')

    def test_predictor_binary_output_wise(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE) \
            .print_predictions() \
            .print_ground_truth()
        CmdRunner(builder).run('predictor-binary-output-wise')

    def test_predictor_binary_output_wise_based_on_probabilities(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .store_evaluation(False) \
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_OUTPUT_WISE_BASED_ON_PROBABILITIES) \
            .print_predictions() \
            .print_ground_truth() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-binary-output-wise_based-on-probabilities')

    def test_predictor_binary_output_wise_incremental(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('predictor-binary-output-wise_incremental')

    def test_predictor_binary_output_wise_incremental_based_on_probabilities(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_OUTPUT_WISE_BASED_ON_PROBABILITIES) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-binary-output-wise_incremental_based-on-probabilities')

    def test_predictor_binary_output_wise_sparse(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE) \
            .print_predictions() \
            .print_ground_truth() \
            .prediction_format(SparsePolicy.FORCE_SPARSE)
        CmdRunner(builder).run('predictor-binary-output-wise_sparse')

    def test_predictor_binary_output_wise_sparse_incremental(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE) \
            .prediction_format(SparsePolicy.FORCE_SPARSE) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('predictor-binary-output-wise_sparse_incremental')

    def test_predictor_binary_example_wise(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors()
        CmdRunner(builder).run('predictor-binary-example-wise')

    def test_predictor_binary_example_wise_based_on_probabilities(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .store_evaluation(False) \
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_EXAMPLE_WISE_BASED_ON_PROBABILITIES) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-binary-example-wise_based-on-probabilities')

    def test_predictor_binary_example_wise_incremental(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('predictor-binary-example-wise_incremental')

    def test_predictor_binary_example_wise_incremental_based_on_probabilities(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_EXAMPLE_WISE_BASED_ON_PROBABILITIES) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-binary-example-wise_incremental_based-on-probabilities')

    def test_predictor_binary_example_wise_sparse(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors() \
            .prediction_format(SparsePolicy.FORCE_SPARSE)
        CmdRunner(builder).run('predictor-binary-example-wise_sparse')

    def test_predictor_binary_example_wise_sparse_incremental(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE) \
            .prediction_format(SparsePolicy.FORCE_SPARSE) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('predictor-binary-example-wise_sparse_incremental')

    def test_predictor_binary_gfm(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .store_evaluation(False) \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_GFM) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-binary-gfm')

    def test_predictor_binary_gfm_incremental(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_GFM) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-binary-gfm_incremental')

    def test_predictor_binary_gfm_sparse(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .store_evaluation(False) \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_GFM) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors() \
            .prediction_format(SparsePolicy.FORCE_SPARSE) \
            .set_model_dir()
        CmdRunner(builder).run('predictor-binary-gfm_sparse')

    def test_predictor_binary_gfm_sparse_incremental(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .binary_predictor(BinaryPredictorParameter.BINARY_PREDICTOR_GFM) \
            .prediction_format(SparsePolicy.FORCE_SPARSE) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-binary-gfm_sparse_incremental')

    def test_predictor_score_output_wise(self):
        builder = self._create_cmd_builder() \
            .prediction_type(PredictionType.SCORES) \
            .print_predictions() \
            .print_ground_truth()
        CmdRunner(builder).run('predictor-score-output-wise')

    def test_predictor_score_output_wise_incremental(self):
        builder = self._create_cmd_builder() \
            .prediction_type(PredictionType.SCORES) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('predictor-score-output-wise_incremental')

    def test_predictor_probability_output_wise(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .store_evaluation(False) \
            .prediction_type(PredictionType.PROBABILITIES) \
            .probability_predictor(ProbabilityPredictorParameter.PROBABILITY_PREDICTOR_OUTPUT_WISE) \
            .print_predictions() \
            .print_ground_truth() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-probability-output-wise')

    def test_predictor_probability_output_wise_incremental(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .prediction_type(PredictionType.PROBABILITIES) \
            .probability_predictor(ProbabilityPredictorParameter.PROBABILITY_PREDICTOR_OUTPUT_WISE) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-probability-output-wise_incremental')

    def test_predictor_probability_marginalized(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .store_evaluation(False) \
            .prediction_type(PredictionType.PROBABILITIES) \
            .probability_predictor(ProbabilityPredictorParameter.PROBABILITY_PREDICTOR_MARGINALIZED) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-probability-marginalized')

    def test_predictor_probability_marginalized_incremental(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .prediction_type(PredictionType.PROBABILITIES) \
            .probability_predictor(ProbabilityPredictorParameter.PROBABILITY_PREDICTOR_MARGINALIZED) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-probability-marginalized_incremental')

    def test_statistics_sparse_output_format_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.numerical_sparse) \
            .sparse_statistic_format() \
            .output_format(SparsePolicy.FORCE_DENSE) \
            .default_rule(False) \
            .loss(ClassificationLossParameter.LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .head_type(HeadTypeParameter.HEAD_TYPE_SINGLE)
        CmdRunner(builder).run('statistics-sparse_output-format-dense')

    def test_statistics_sparse_output_format_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.numerical_sparse) \
            .sparse_statistic_format() \
            .output_format(SparsePolicy.FORCE_SPARSE) \
            .default_rule(False) \
            .loss(ClassificationLossParameter.LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .head_type(HeadTypeParameter.HEAD_TYPE_SINGLE)
        CmdRunner(builder).run('statistics-sparse_output-format-sparse')

    @pytest.mark.parametrize('head_type, label_binning', [
        (HeadTypeParameter.HEAD_TYPE_SINGLE, None),
        (HeadTypeParameter.HEAD_TYPE_COMPLETE, None),
        (HeadTypeParameter.HEAD_TYPE_COMPLETE, BINNING_EQUAL_WIDTH),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED, None),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED, BINNING_EQUAL_WIDTH),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC, None),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC, BINNING_EQUAL_WIDTH),
    ])
    @pytest.mark.parametrize('statistic_type', [
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT32,
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT64,
    ])
    def test_decomposable_head_type(self, head_type: str, label_binning: Optional[str], statistic_type: str):
        builder = self._create_cmd_builder() \
            .loss(ClassificationLossParameter.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(statistic_type) \
            .head_type(head_type) \
            .label_binning(label_binning) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'decomposable-{head_type}-heads'
                               + (f'_{label_binning}-label-binning' if label_binning else '')
                               + f'_{statistic_type}-statistics')

    @pytest.mark.parametrize('head_type, label_binning', [
        (HeadTypeParameter.HEAD_TYPE_SINGLE, None),
        (HeadTypeParameter.HEAD_TYPE_COMPLETE, NONE),
        (HeadTypeParameter.HEAD_TYPE_COMPLETE, BINNING_EQUAL_WIDTH),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED, NONE),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED, BINNING_EQUAL_WIDTH),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC, NONE),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC, BINNING_EQUAL_WIDTH),
    ])
    @pytest.mark.parametrize('statistic_type', [
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT32,
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT64,
    ])
    def test_non_decomposable_head_type(self, head_type: str, label_binning: Optional[str], statistic_type: str):
        builder = self._create_cmd_builder() \
            .loss(ClassificationLossParameter.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(statistic_type) \
            .head_type(head_type) \
            .label_binning(label_binning) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'non-decomposable-{head_type}-heads'
                               + (f'_{label_binning}-label-binning' if label_binning else '')
                               + f'_{statistic_type}-statistics')

    @pytest.mark.parametrize('global_pruning', [
        GlobalPruningParameter.GLOBAL_PRUNING_POST,
        GlobalPruningParameter.GLOBAL_PRUNING_PRE,
    ])
    @pytest.mark.parametrize('holdout', [
        SAMPLING_STRATIFIED_OUTPUT_WISE,
        SAMPLING_STRATIFIED_EXAMPLE_WISE,
    ])
    def test_global_pruning_stratified_holdout(self, global_pruning: str, holdout: str):
        builder = self._create_cmd_builder() \
            .global_pruning(global_pruning) \
            .holdout(holdout) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'{global_pruning}_{holdout}-holdout')
