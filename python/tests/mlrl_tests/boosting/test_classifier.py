"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from typing import Any

import pytest

from ..common.cmd_builder_classification import ClassificationCmdBuilder
from ..common.cmd_runner import CmdRunner
from ..common.datasets import Dataset
from ..common.integration_tests_classification import ClassificationIntegrationTests
from .cmd_builder import BoomerCmdBuilderMixin
from .cmd_builder_classification import BoomerClassifierCmdBuilder
from .integration_tests import BoomerIntegrationTestsMixin


@pytest.mark.boosting
@pytest.mark.classification
class TestBoomerClassifier(ClassificationIntegrationTests, BoomerIntegrationTestsMixin):
    """
    Defines a series of integration tests for the BOOMER algorithm for classification problems.
    """

    def _create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        return BoomerClassifierCmdBuilder(dataset=dataset)

    def test_single_label_scores(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.single_output) \
            .prediction_type(ClassificationCmdBuilder.PREDICTION_TYPE_SCORES) \
            .print_evaluation()
        CmdRunner(builder).run('single-label-scores')

    def test_single_label_probabilities(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.single_output) \
            .prediction_type(ClassificationCmdBuilder.PREDICTION_TYPE_PROBABILITIES) \
            .print_evaluation()
        CmdRunner(builder).run('single-label-probabilities')

    def test_loss_logistic_decomposable_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32)
        CmdRunner(builder).run('loss-logistic-decomposable_32-bit-statistics')

    def test_loss_logistic_decomposable_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64)
        CmdRunner(builder).run('loss-logistic-decomposable_64-bit-statistics')

    def test_loss_logistic_non_decomposable_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32)
        CmdRunner(builder).run('loss-logistic-non-decomposable_32-bit-statistics')

    def test_loss_logistic_non_decomposable_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64)
        CmdRunner(builder).run('loss-logistic-non-decomposable_64-bit-statistics')

    def test_loss_squared_hinge_decomposable_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32)
        CmdRunner(builder).run('loss-squared-hinge-decomposable_32-bit-statistics')

    def test_loss_squared_hinge_decomposable_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64)
        CmdRunner(builder).run('loss-squared-hinge-decomposable_64-bit-statistics')

    def test_loss_squared_hinge_non_decomposable_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_SQUARED_HINGE_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32)
        CmdRunner(builder).run('loss-squared-hinge-non-decomposable_32-bit-statistics')

    def test_loss_squared_hinge_non_decomposable_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_SQUARED_HINGE_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64)
        CmdRunner(builder).run('loss-squared-hinge-non-decomposable_64-bit-statistics')

    def test_predictor_binary_output_wise(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_OUTPUT_WISE) \
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
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_OUTPUT_WISE) \
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
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_OUTPUT_WISE) \
            .print_predictions() \
            .print_ground_truth() \
            .sparse_prediction_format()
        CmdRunner(builder).run('predictor-binary-output-wise_sparse')

    def test_predictor_binary_output_wise_sparse_incremental(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_OUTPUT_WISE) \
            .sparse_prediction_format() \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('predictor-binary-output-wise_sparse_incremental')

    def test_predictor_binary_example_wise(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_EXAMPLE_WISE) \
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
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_EXAMPLE_WISE) \
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
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_EXAMPLE_WISE) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors() \
            .sparse_prediction_format()
        CmdRunner(builder).run('predictor-binary-example-wise_sparse')

    def test_predictor_binary_example_wise_sparse_incremental(self):
        builder = self._create_cmd_builder() \
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_EXAMPLE_WISE) \
            .sparse_prediction_format() \
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
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_GFM) \
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
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_GFM) \
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
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_GFM) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors() \
            .sparse_prediction_format() \
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
            .binary_predictor(BoomerClassifierCmdBuilder.BINARY_PREDICTOR_GFM) \
            .sparse_prediction_format() \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-binary-gfm_sparse_incremental')

    def test_predictor_score_output_wise(self):
        builder = self._create_cmd_builder() \
            .prediction_type(ClassificationCmdBuilder.PREDICTION_TYPE_SCORES) \
            .print_predictions() \
            .print_ground_truth()
        CmdRunner(builder).run('predictor-score-output-wise')

    def test_predictor_score_output_wise_incremental(self):
        builder = self._create_cmd_builder() \
            .prediction_type(ClassificationCmdBuilder.PREDICTION_TYPE_SCORES) \
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
            .prediction_type(ClassificationCmdBuilder.PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(BoomerClassifierCmdBuilder.PROBABILITY_PREDICTOR_OUTPUT_WISE) \
            .print_predictions() \
            .print_ground_truth() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-probability-output-wise')

    def test_predictor_probability_output_wise_incremental(self):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .prediction_type(ClassificationCmdBuilder.PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(BoomerClassifierCmdBuilder.PROBABILITY_PREDICTOR_OUTPUT_WISE) \
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
            .prediction_type(ClassificationCmdBuilder.PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(BoomerClassifierCmdBuilder.PROBABILITY_PREDICTOR_MARGINALIZED) \
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
            .prediction_type(ClassificationCmdBuilder.PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(BoomerClassifierCmdBuilder.PROBABILITY_PREDICTOR_MARGINALIZED) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        CmdRunner(builder).run('predictor-probability-marginalized_incremental')

    def test_statistics_sparse_output_format_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.numerical_sparse) \
            .sparse_statistic_format() \
            .sparse_output_format(False) \
            .default_rule(False) \
            .loss(BoomerClassifierCmdBuilder.LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_SINGLE)
        CmdRunner(builder).run('statistics-sparse_output-format-dense')

    def test_statistics_sparse_output_format_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.numerical_sparse) \
            .sparse_statistic_format() \
            .sparse_output_format() \
            .default_rule(False) \
            .loss(BoomerClassifierCmdBuilder.LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_SINGLE)
        CmdRunner(builder).run('statistics-sparse_output-format-sparse')

    def test_decomposable_single_output_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-single-output-heads_32-bit-statistics')

    def test_decomposable_single_output_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-single-output-heads_64-bit-statistics')

    def test_decomposable_complete_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-complete-heads_32-bit-statistics')

    def test_decomposable_complete_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-complete-heads_64-bit-statistics')

    def test_decomposable_complete_heads_equal_width_label_binning_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_COMPLETE) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-complete-heads_equal-width-label-binning_32-bit-statistics')

    def test_decomposable_complete_heads_equal_width_label_binning_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_COMPLETE) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-complete-heads_equal-width-label-binning_64-bit-statistics')

    def test_decomposable_partial_fixed_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-fixed-heads_32-bit-statistics')

    def test_decomposable_partial_fixed_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-fixed-heads_64-bit-statistics')

    def test_decomposable_partial_fixed_heads_equal_width_label_binning_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-fixed-heads_equal-width-label-binning_32-bit-statistics')

    def test_decomposable_partial_fixed_heads_equal_width_label_binning_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-fixed-heads_equal-width-label-binning_64-bit-statistics')

    def test_decomposable_partial_dynamic_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-dynamic-heads_32-bit-statistics')

    def test_decomposable_partial_dynamic_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-dynamic-heads_64-bit-statistics')

    def test_decomposable_partial_dynamic_heads_equal_width_label_binning_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-dynamic-heads_equal-width-label-binning_32-bit-statistics')

    def test_decomposable_partial_dynamic_heads_equal_width_label_binning_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-dynamic-heads_equal-width-label-binning_64-bit-statistics')

    def test_non_decomposable_single_label_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-single-output-heads_32-bit-statistics')

    def test_non_decomposable_single_label_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-single-output-heads_64-bit-statistics')

    def test_non_decomposable_complete_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_COMPLETE) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_NO) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-complete-heads_32-bit-statistics')

    def test_non_decomposable_complete_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_COMPLETE) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_NO) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-complete-heads_64-bit-statistics')

    def test_non_decomposable_complete_heads_equal_width_label_binning_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_COMPLETE) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-complete-heads_equal-width-label-binning_32-bit-statistics')

    def test_non_decomposable_complete_heads_equal_width_label_binning_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_COMPLETE) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-complete-heads_equal-width-label-binning_64-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_NO) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-fixed-heads_32-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_NO) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-fixed-heads_64-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_equal_width_label_binning_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-fixed-heads_equal-width-label-binning_32-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_equal_width_label_binning_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-fixed-heads_equal-width-label-binning_64-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_NO) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-dynamic-heads_32-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_NO) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-dynamic-heads_64-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_equal_width_label_binning_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-dynamic-heads_equal-width-label-binning_32-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_equal_width_label_binning_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(BoomerClassifierCmdBuilder.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64) \
            .head_type(BoomerCmdBuilderMixin.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(BoomerClassifierCmdBuilder.LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-dynamic-heads_equal-width-label-binning_64-bit-statistics')

    def test_global_post_pruning_stratified_output_wise_holdout(self):
        builder = self._create_cmd_builder() \
            .global_pruning(BoomerCmdBuilderMixin.GLOBAL_PRUNING_POST) \
            .holdout(ClassificationCmdBuilder.HOLDOUT_STRATIFIED_OUTPUT_WISE) \
            .print_model_characteristics()
        CmdRunner(builder).run('post-pruning_stratified-output-wise-holdout')

    def test_global_post_pruning_stratified_example_wise_holdout(self):
        builder = self._create_cmd_builder() \
            .global_pruning(BoomerCmdBuilderMixin.GLOBAL_PRUNING_POST) \
            .holdout(ClassificationCmdBuilder.HOLDOUT_STRATIFIED_EXAMPLE_WISE) \
            .print_model_characteristics()
        CmdRunner(builder).run('post-pruning_stratified-example-wise-holdout')

    def test_global_pre_pruning_stratified_output_wise_holdout(self):
        builder = self._create_cmd_builder() \
            .global_pruning(BoomerCmdBuilderMixin.GLOBAL_PRUNING_PRE) \
            .holdout(ClassificationCmdBuilder.HOLDOUT_STRATIFIED_OUTPUT_WISE) \
            .print_model_characteristics()
        CmdRunner(builder).run('pre-pruning_stratified-output-wise-holdout')

    def test_global_pre_pruning_stratified_example_wise_holdout(self):
        builder = self._create_cmd_builder() \
            .global_pruning(BoomerCmdBuilderMixin.GLOBAL_PRUNING_PRE) \
            .holdout(ClassificationCmdBuilder.HOLDOUT_STRATIFIED_EXAMPLE_WISE) \
            .print_model_characteristics()
        CmdRunner(builder).run('pre-pruning_stratified-example-wise-holdout')
