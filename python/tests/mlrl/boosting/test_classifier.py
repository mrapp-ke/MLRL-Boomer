"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Any

from ..common.cmd_builder import DATASET_EMOTIONS
from ..common.cmd_builder_classification import HOLDOUT_STRATIFIED_EXAMPLE_WISE, HOLDOUT_STRATIFIED_OUTPUT_WISE, \
    PREDICTION_TYPE_PROBABILITIES, PREDICTION_TYPE_SCORES
from ..common.integration_tests_classification import ClassificationIntegrationTests
from .cmd_builder import GLOBAL_PRUNING_POST, GLOBAL_PRUNING_PRE, HEAD_TYPE_COMPLETE, HEAD_TYPE_PARTIAL_DYNAMIC, \
    HEAD_TYPE_PARTIAL_FIXED, HEAD_TYPE_SINGLE, STATISTIC_TYPE_FLOAT32, STATISTIC_TYPE_FLOAT64
from .cmd_builder_classification import BINARY_PREDICTOR_EXAMPLE_WISE, \
    BINARY_PREDICTOR_EXAMPLE_WISE_BASED_ON_PROBABILITIES, BINARY_PREDICTOR_GFM, BINARY_PREDICTOR_OUTPUT_WISE, \
    BINARY_PREDICTOR_OUTPUT_WISE_BASED_ON_PROBABILITIES, LABEL_BINNING_EQUAL_WIDTH, LABEL_BINNING_NO, \
    LOSS_LOGISTIC_DECOMPOSABLE, LOSS_LOGISTIC_NON_DECOMPOSABLE, LOSS_SQUARED_HINGE_DECOMPOSABLE, \
    LOSS_SQUARED_HINGE_NON_DECOMPOSABLE, PROBABILITY_PREDICTOR_MARGINALIZED, PROBABILITY_PREDICTOR_OUTPUT_WISE, \
    BoomerClassifierCmdBuilder
from .integration_tests import BoomerIntegrationTestsMixin


class BoomerClassifierIntegrationTests(ClassificationIntegrationTests, BoomerIntegrationTestsMixin):
    """
    Defines a series of integration tests for the BOOMER algorithm for classification problems.
    """

    # pylint: disable=invalid-name
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def _create_cmd_builder(self, dataset: str = DATASET_EMOTIONS) -> Any:
        return BoomerClassifierCmdBuilder(self, dataset=dataset)

    def test_single_label_scores(self):
        """
        Tests the evaluation of the rule learning algorithm when predicting scores for a single-label problem.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_single_output) \
            .prediction_type(PREDICTION_TYPE_SCORES) \
            .print_evaluation()
        builder.run_cmd('single-label-scores')

    def test_single_label_probabilities(self):
        """
        Tests the evaluation of the rule learning algorithm when predicting probabilities for a single-label problem.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_single_output) \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .print_evaluation()
        builder.run_cmd('single-label-probabilities')

    def test_loss_logistic_decomposable_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the decomposable logistic loss function and 32-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32)
        builder.run_cmd('loss-logistic-decomposable_32-bit-statistics')

    def test_loss_logistic_decomposable_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the decomposable logistic loss function and 64-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64)
        builder.run_cmd('loss-logistic-decomposable_64-bit-statistics')

    def test_loss_logistic_non_decomposable_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the non-decomposable logistic loss function and 32-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32)
        builder.run_cmd('loss-logistic-non-decomposable_32-bit-statistics')

    def test_loss_logistic_non_decomposable_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the non-decomposable logistic loss function and 64-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64)
        builder.run_cmd('loss-logistic-non-decomposable_64-bit-statistics')

    def test_loss_squared_hinge_decomposable_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the decomposable squared hinge loss function and 32-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32)
        builder.run_cmd('loss-squared-hinge-decomposable_32-bit-statistics')

    def test_loss_squared_hinge_decomposable_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the decomposable squared hinge loss function and 64-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64)
        builder.run_cmd('loss-squared-hinge-decomposable_64-bit-statistics')

    def test_loss_squared_hinge_non_decomposable_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the non-decomposable squared hinge loss function and 32-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_HINGE_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32)
        builder.run_cmd('loss-squared-hinge-non-decomposable_32-bit-statistics')

    def test_loss_squared_hinge_non_decomposable_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the non-decomposable squared hinge loss function and 64-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_HINGE_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64)
        builder.run_cmd('loss-squared-hinge-non-decomposable_64-bit-statistics')

    def test_predictor_binary_output_wise(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained for each label individually.
        """
        builder = self._create_cmd_builder() \
            .binary_predictor(BINARY_PREDICTOR_OUTPUT_WISE) \
            .print_predictions()
        builder.run_cmd('predictor-binary-output-wise')

    def test_predictor_binary_output_wise_based_on_probabilities(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained for each label individually based on
        probability estimates.
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .binary_predictor(BINARY_PREDICTOR_OUTPUT_WISE_BASED_ON_PROBABILITIES) \
            .print_predictions() \
            .set_model_dir()
        builder.run_cmd('predictor-binary-output-wise_based-on-probabilities')

    def test_predictor_binary_output_wise_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting binary labels
        that are obtained for each label individually.
        """
        builder = self._create_cmd_builder() \
            .binary_predictor(BINARY_PREDICTOR_OUTPUT_WISE) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('predictor-binary-output-wise_incremental')

    def test_predictor_binary_output_wise_incremental_based_on_probabilities(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting binary labels
        that are obtained for each label individually based on probability estimates.
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .binary_predictor(BINARY_PREDICTOR_OUTPUT_WISE_BASED_ON_PROBABILITIES) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        builder.run_cmd('predictor-binary-output-wise_incremental_based-on-probabilities')

    def test_predictor_binary_output_wise_sparse(self):
        """
        Tests the BOOMER algorithm when predicting sparse binary labels that are obtained for each label individually.
        """
        builder = self._create_cmd_builder() \
            .binary_predictor(BINARY_PREDICTOR_OUTPUT_WISE) \
            .print_predictions() \
            .sparse_prediction_format()
        builder.run_cmd('predictor-binary-output-wise_sparse')

    def test_predictor_binary_output_wise_sparse_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting sparse binary
        labels that are obtained for each label individually.
        """
        builder = self._create_cmd_builder() \
            .binary_predictor(BINARY_PREDICTOR_OUTPUT_WISE) \
            .sparse_prediction_format() \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('predictor-binary-output-wise_sparse_incremental')

    def test_predictor_binary_example_wise(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained by predicting one of the known label
        vectors.
        """
        builder = self._create_cmd_builder() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE) \
            .print_predictions() \
            .print_label_vectors()
        builder.run_cmd('predictor-binary-example-wise')

    def test_predictor_binary_example_wise_based_on_probabilities(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained by predicting one of the known label
        vectors based on probability estimates.
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE_BASED_ON_PROBABILITIES) \
            .print_predictions() \
            .print_label_vectors() \
            .set_model_dir()
        builder.run_cmd('predictor-binary-example-wise_based-on-probabilities')

    def test_predictor_binary_example_wise_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting one of the
        known label vectors.
        """
        builder = self._create_cmd_builder() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('predictor-binary-example-wise_incremental')

    def test_predictor_binary_example_wise_incremental_based_on_probabilities(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting one of the
        known label vectors based on probability estimates.
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE_BASED_ON_PROBABILITIES) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        builder.run_cmd('predictor-binary-example-wise_incremental_based-on-probabilities')

    def test_predictor_binary_example_wise_sparse(self):
        """
        Tests the BOOMER algorithm when predicting sparse binary labels that are obtained by predicting one of the known
        label vectors.
        """
        builder = self._create_cmd_builder() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE) \
            .print_predictions() \
            .print_label_vectors() \
            .sparse_prediction_format()
        builder.run_cmd('predictor-binary-example-wise_sparse')

    def test_predictor_binary_example_wise_sparse_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting one of the
        known label vectors.
        """
        builder = self._create_cmd_builder() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE) \
            .sparse_prediction_format() \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('predictor-binary-example-wise_sparse_incremental')

    def test_predictor_binary_gfm(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained via the general F-measure maximizer
        (GFM).
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .binary_predictor(BINARY_PREDICTOR_GFM) \
            .print_predictions() \
            .print_label_vectors() \
            .set_model_dir()
        builder.run_cmd('predictor-binary-gfm')

    def test_predictor_binary_gfm_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting binary labels
        that are obtained via the general F-measure maximizer (GFM).
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .binary_predictor(BINARY_PREDICTOR_GFM) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        builder.run_cmd('predictor-binary-gfm_incremental')

    def test_predictor_binary_gfm_sparse(self):
        """
        Tests the BOOMER algorithm when predicting sparse binary labels that are obtained via the general F-measure
        maximizer (GFM).
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .binary_predictor(BINARY_PREDICTOR_GFM) \
            .print_predictions() \
            .print_label_vectors() \
            .sparse_prediction_format() \
            .set_model_dir()
        builder.run_cmd('predictor-binary-gfm_sparse')

    def test_predictor_binary_gfm_sparse_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting sparse binary
        labels that are obtained via the general F-measure maximizer (GFM).
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .binary_predictor(BINARY_PREDICTOR_GFM) \
            .sparse_prediction_format() \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        builder.run_cmd('predictor-binary-gfm_sparse_incremental')

    def test_predictor_score_output_wise(self):
        """
        Tests the BOOMER algorithm when predicting scores that are obtained in an output-wise manner.
        """
        builder = self._create_cmd_builder() \
            .prediction_type(PREDICTION_TYPE_SCORES) \
            .print_predictions()
        builder.run_cmd('predictor-score-output-wise')

    def test_predictor_score_output_wise_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting scores that are
        obtained in an output-wise manner.
        """
        builder = self._create_cmd_builder() \
            .prediction_type(PREDICTION_TYPE_SCORES) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('predictor-score-output-wise_incremental')

    def test_predictor_probability_output_wise(self):
        """
        Tests the BOOMER algorithm when predicting probabilities that are obtained by applying a transformation function
        to each output.
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(PROBABILITY_PREDICTOR_OUTPUT_WISE) \
            .print_predictions() \
            .set_model_dir()
        builder.run_cmd('predictor-probability-output-wise')

    def test_predictor_probability_output_wise_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting probabilities
        that are obtained by applying a transformation function to each output.
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(PROBABILITY_PREDICTOR_OUTPUT_WISE) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        builder.run_cmd('predictor-probability-output-wise_incremental')

    def test_predictor_probability_marginalized(self):
        """
        Tests the BOOMER algorithm when predicting probabilities that are obtained via marginalization over the known
        label vectors.
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(PROBABILITY_PREDICTOR_MARGINALIZED) \
            .print_predictions() \
            .print_label_vectors() \
            .set_model_dir()
        builder.run_cmd('predictor-probability-marginalized')

    def test_predictor_probability_marginalized_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting probabilities
        that are obtained via marginalization over the known label vectors.
        """
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(PROBABILITY_PREDICTOR_MARGINALIZED) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        builder.run_cmd('predictor-probability-marginalized_incremental')

    def test_statistics_sparse_output_format_dense(self):
        """
        Tests the BOOMER algorithm when using sparse data structures for storing the statistics and a dense output
        matrix.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_numerical_sparse) \
            .sparse_statistic_format() \
            .sparse_output_format(False) \
            .default_rule(False) \
            .loss(LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_SINGLE)
        builder.run_cmd('statistics-sparse_output-format-dense')

    def test_statistics_sparse_output_format_sparse(self):
        """
        Tests the BOOMER algorithm when using sparse data structures for storing the statistics and a sparse output
        matrix.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_numerical_sparse) \
            .sparse_statistic_format() \
            .sparse_output_format() \
            .default_rule(False) \
            .loss(LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_SINGLE)
        builder.run_cmd('statistics-sparse_output-format-sparse')

    def test_decomposable_single_output_heads_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 32-bit statistics for the induction of
        rules with single-output heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-single-output-heads_32-bit-statistics')

    def test_decomposable_single_output_heads_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 64-bit statistics for the induction of
        rules with single-output heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-single-output-heads_64-bit-statistics')

    def test_decomposable_complete_heads_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 32-bit statistics for the induction of
        rules with complete heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-complete-heads_32-bit-statistics')

    def test_decomposable_complete_heads_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 64-bit statistics for the induction of
        rules with complete heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-complete-heads_64-bit-statistics')

    def test_decomposable_complete_heads_equal_width_label_binning_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 32-bit statistics and equal-width label
        binning for the induction of rules with complete heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-complete-heads_equal-width-label-binning_32-bit-statistics')

    def test_decomposable_complete_heads_equal_width_label_binning_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 64-bit statistics and equal-width label
        binning for the induction of rules with complete heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-complete-heads_equal-width-label-binning_64-bit-statistics')

    def test_decomposable_partial_fixed_heads_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 32-bit statistics for the induction of
        rules that predict for a number of labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-fixed-heads_32-bit-statistics')

    def test_decomposable_partial_fixed_heads_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 64-bit statistics for the induction of
        rules that predict for a number of labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-fixed-heads_64-bit-statistics')

    def test_decomposable_partial_fixed_heads_equal_width_label_binning_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function, 32-bit statistics and equal-width label
        binning for the induction of rules that predict for a number of labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-fixed-heads_equal-width-label-binning_32-bit-statistics')

    def test_decomposable_partial_fixed_heads_equal_width_label_binning_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function, 64-bit statistics and equal-width label
        binning for the induction of rules that predict for a number of labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-fixed-heads_equal-width-label-binning_64-bit-statistics')

    def test_decomposable_partial_dynamic_heads_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 32-bit statistics for the induction of
        rules that predict for a dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-dynamic-heads_32-bit-statistics')

    def test_decomposable_partial_dynamic_heads_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function and 64-bit statistics for the induction of
        rules that predict for a dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-dynamic-heads_64-bit-statistics')

    def test_decomposable_partial_dynamic_heads_equal_width_label_binning_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function, 32-bit statistics and equal-width label
        binning for the induction of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-dynamic-heads_equal-width-label-binning_32-bit-statistics')

    def test_decomposable_partial_dynamic_heads_equal_width_label_binning_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function, 64-bit statistics and equal-width label
        binning for the induction of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-dynamic-heads_equal-width-label-binning_64-bit-statistics')

    def test_non_decomposable_single_label_heads_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and 32-bit statistics for the induction
        of rules with single-output heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-single-output-heads_32-bit-statistics')

    def test_non_decomposable_single_label_heads_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and 64-bit statistics for the induction
        of rules with single-output heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-single-output-heads_64-bit-statistics')

    def test_non_decomposable_complete_heads_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and 32-bit statistics for the induction
        of rules with complete heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_NO) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-complete-heads_32-bit-statistics')

    def test_non_decomposable_complete_heads_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and 64-bit statistics for the induction
        of rules with complete heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_NO) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-complete-heads_64-bit-statistics')

    def test_non_decomposable_complete_heads_equal_width_label_binning_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function, 32-bit statistics and equal-width label
        binning for the induction of rules with complete heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-complete-heads_equal-width-label-binning_32-bit-statistics')

    def test_non_decomposable_complete_heads_equal_width_label_binning_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function, 64-bit statistics and equal-width label
        binning for the induction of rules with complete heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-complete-heads_equal-width-label-binning_64-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and 32-bit statistics for the induction
        of rules that predict for a number of labels
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_NO) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-fixed-heads_32-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and 64-bit statistics for the induction
        of rules that predict for a number of labels
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_NO) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-fixed-heads_64-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_equal_width_label_binning_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function, 32-bit statistics and equal-width label
        binning for the induction of rules that predict for a number of labels
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-fixed-heads_equal-width-label-binning_32-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_equal_width_label_binning_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function, 64-bit statistics and equal-width label
        binning for the induction of rules that predict for a number of labels
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-fixed-heads_equal-width-label-binning_64-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and 32-bit statistics for the induction
        of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_NO) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-dynamic-heads_32-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and 64-bit statistics for the induction
        of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_NO) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-dynamic-heads_64-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_equal_width_label_binning_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function, 32-bit statistics and equal-width label
        binning for the induction of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-dynamic-heads_equal-width-label-binning_32-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_equal_width_label_binning_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function, 64-bit statistics and equal-width label
        binning for the induction of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT64) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-dynamic-heads_equal-width-label-binning_64-bit-statistics')

    def test_global_post_pruning_stratified_output_wise_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via label-wise stratified sampling for
        global post-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(GLOBAL_PRUNING_POST) \
            .holdout(HOLDOUT_STRATIFIED_OUTPUT_WISE) \
            .print_model_characteristics()
        builder.run_cmd('post-pruning_stratified-output-wise-holdout')

    def test_global_post_pruning_stratified_example_wise_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via example-wise stratified sampling for
        global post-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(GLOBAL_PRUNING_POST) \
            .holdout(HOLDOUT_STRATIFIED_EXAMPLE_WISE) \
            .print_model_characteristics()
        builder.run_cmd('post-pruning_stratified-example-wise-holdout')

    def test_global_pre_pruning_stratified_output_wise_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via label-wise stratified sampling for
        global pre-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(GLOBAL_PRUNING_PRE) \
            .holdout(HOLDOUT_STRATIFIED_OUTPUT_WISE) \
            .print_model_characteristics()
        builder.run_cmd('pre-pruning_stratified-output-wise-holdout')

    def test_global_pre_pruning_stratified_example_wise_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via example-wise stratified sampling for
        global pre-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(GLOBAL_PRUNING_PRE) \
            .holdout(HOLDOUT_STRATIFIED_EXAMPLE_WISE) \
            .print_model_characteristics()
        builder.run_cmd('pre-pruning_stratified-example-wise-holdout')
