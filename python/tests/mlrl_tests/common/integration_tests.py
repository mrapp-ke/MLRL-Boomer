"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
import os

from abc import ABC
from typing import Any, Optional

import pytest

from .cmd_builder import CmdBuilder
from .cmd_runner import CmdRunner
from .datasets import Dataset

ci_only = pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') != 'true', reason='Disabled unless run on CI')


class IntegrationTests(ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm.
    """

    def _create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        """
        Must be implemented by subclasses in order to create an object of type `CmdBuilder` that allows to configure the
        command for running a rule learner.

        :param dataset: The dataset that should be used
        :return:        The object that has been created
        """
        raise NotImplementedError('Method _create_cmd_builder not implemented by test class')

    @ci_only
    def test_help(self):
        builder = self._create_cmd_builder() \
            .set_show_help()
        CmdRunner(builder).run('help')

    def test_single_output(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.single_output) \
            .print_evaluation()
        CmdRunner(builder).run('single-output')

    def test_sparse_feature_value(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.numerical_sparse) \
            .sparse_feature_value(1.0)
        CmdRunner(builder).run('sparse-feature-value')

    def test_meka_format(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.meka) \
            .print_evaluation(False)
        CmdRunner(builder).run('meka-format')

    def test_evaluation_no_data_split(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .no_data_split() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_no-data-split')

    def test_evaluation_train_test(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_train-test')

    def test_evaluation_train_test_predefined(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default + '-predefined') \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_train-test-predefined')

    def test_evaluation_cross_validation(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_cross-validation')

    def test_evaluation_cross_validation_predefined(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default + '-predefined') \
            .cross_validation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_cross-validation-predefined')

    def test_evaluation_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation(current_fold=1) \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_single-fold')

    def test_evaluation_training_data(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .predict_for_training_data() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_training-data')

    def test_evaluation_incremental(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_incremental')

    def test_model_persistence_train_test(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .set_model_dir()
        CmdRunner(builder).run('model-persistence_train-test')

    def test_model_persistence_cross_validation(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation() \
            .set_model_dir()
        CmdRunner(builder).run('model-persistence_cross-validation')

    def test_model_persistence_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation(current_fold=1) \
            .set_model_dir()
        CmdRunner(builder).run('model-persistence_single-fold')

    def test_predictions_train_test(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_predictions() \
            .print_ground_truth() \
            .store_predictions() \
            .store_ground_truth()
        CmdRunner(builder).run('predictions_train-test')

    def test_predictions_cross_validation(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_predictions() \
            .print_ground_truth() \
            .store_predictions() \
            .store_ground_truth()
        CmdRunner(builder).run('predictions_cross-validation')

    def test_predictions_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_predictions() \
            .print_ground_truth() \
            .store_predictions() \
            .store_ground_truth()
        CmdRunner(builder).run('predictions_single-fold')

    def test_predictions_training_data(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .predict_for_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_predictions() \
            .print_ground_truth() \
            .store_predictions() \
            .store_ground_truth()
        CmdRunner(builder).run('predictions_training-data')

    def test_prediction_characteristics_train_test(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        CmdRunner(builder).run('prediction-characteristics_train-test')

    def test_prediction_characteristics_cross_validation(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        CmdRunner(builder).run('prediction-characteristics_cross-validation')

    def test_prediction_characteristics_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        CmdRunner(builder).run('prediction-characteristics_single-fold')

    def test_prediction_characteristics_training_data(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .predict_for_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        CmdRunner(builder).run('prediction-characteristics_training-data')

    def test_data_characteristics_train_test(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_data_characteristics() \
            .store_data_characteristics()
        CmdRunner(builder).run('data-characteristics_train-test')

    def test_data_characteristics_cross_validation(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_data_characteristics() \
            .store_data_characteristics()
        CmdRunner(builder).run('data-characteristics_cross-validation')

    def test_data_characteristics_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_data_characteristics() \
            .store_data_characteristics()
        CmdRunner(builder).run('data-characteristics_single-fold')

    def test_model_characteristics_train_test(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .store_model_characteristics()
        CmdRunner(builder).run('model-characteristics_train-test')

    def test_model_characteristics_cross_validation(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .store_model_characteristics()
        CmdRunner(builder).run('model-characteristics_cross-validation')

    def test_model_characteristics_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .store_model_characteristics()
        CmdRunner(builder).run('model-characteristics_single-fold')

    def test_rules_train_test(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_rules() \
            .store_rules()
        CmdRunner(builder).run('rules_train-test')

    def test_rules_cross_validation(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_rules() \
            .store_rules()
        CmdRunner(builder).run('rules_cross-validation')

    def test_rules_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_rules() \
            .store_rules()
        CmdRunner(builder).run('rules_single-fold')

    def test_numeric_features_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.numerical_sparse) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('numeric-features-dense')

    def test_numeric_features_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.numerical_sparse) \
            .sparse_feature_format()
        CmdRunner(builder).run('numeric-features-sparse')

    def test_binary_features_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.binary) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('binary-features-dense')

    def test_binary_features_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.binary) \
            .sparse_feature_format()
        CmdRunner(builder).run('binary-features-sparse')

    def test_nominal_features_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.nominal) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('nominal-features-dense')

    def test_nominal_features_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.nominal) \
            .sparse_feature_format()
        CmdRunner(builder).run('nominal-features-sparse')

    def test_ordinal_features_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.ordinal) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('ordinal-features-dense')

    def test_ordinal_features_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.ordinal) \
            .sparse_feature_format()
        CmdRunner(builder).run('ordinal-features-sparse')

    def test_output_format_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .sparse_output_format(False)
        CmdRunner(builder).run('output-format-dense')

    def test_output_format_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .sparse_output_format()
        CmdRunner(builder).run('output-format-sparse')

    def test_prediction_format_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .sparse_prediction_format(False) \
            .print_predictions() \
            .print_ground_truth()
        CmdRunner(builder).run('prediction-format-dense')

    def test_prediction_format_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .sparse_prediction_format() \
            .print_predictions() \
            .print_ground_truth()
        CmdRunner(builder).run('prediction-format-sparse')

    def test_parameters_train_test(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .set_parameter_save_dir() \
            .set_parameter_load_dir()
        CmdRunner(builder).run('parameters_train-test')

    def test_parameters_cross_validation(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .set_parameter_save_dir() \
            .set_parameter_load_dir()
        CmdRunner(builder).run('parameters_cross-validation')

    def test_parameters_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .set_parameter_save_dir() \
            .set_parameter_load_dir()
        CmdRunner(builder).run('parameters_single-fold')

    @pytest.mark.parametrize('instance_sampling', [
        CmdBuilder.INSTANCE_SAMPLING_NO,
        CmdBuilder.INSTANCE_SAMPLING_WITH_REPLACEMENT,
        CmdBuilder.INSTANCE_SAMPLING_WITHOUT_REPLACEMENT,
    ])
    def test_instance_sampling(self, instance_sampling: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .instance_sampling(instance_sampling)
        CmdRunner(builder).run(f'instance-sampling-{instance_sampling}')

    @pytest.mark.parametrize('feature_sampling', [
        CmdBuilder.FEATURE_SAMPLING_NO,
        CmdBuilder.FEATURE_SAMPLING_WITHOUT_REPLACEMENT,
    ])
    def test_feature_sampling(self, feature_sampling: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .feature_sampling(feature_sampling)
        CmdRunner(builder).run(f'feature-sampling-{feature_sampling}')

    @pytest.mark.parametrize('output_sampling', [
        CmdBuilder.OUTPUT_SAMPLING_NO,
        CmdBuilder.OUTPUT_SAMPLING_ROUND_ROBIN,
        CmdBuilder.OUTPUT_SAMPLING_WITHOUT_REPLACEMENT,
    ])
    def test_output_sampling(self, output_sampling: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .output_sampling(output_sampling)
        CmdRunner(builder).run(f'output-sampling-{output_sampling}')

    @pytest.mark.parametrize('rule_pruning, instance_sampling', [
        (CmdBuilder.RULE_PRUNING_NO, None),
        (CmdBuilder.RULE_PRUNING_IREP, CmdBuilder.INSTANCE_SAMPLING_WITHOUT_REPLACEMENT),
    ])
    def test_rule_pruning(self, rule_pruning: str, instance_sampling: Optional[str], dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .instance_sampling(instance_sampling) \
            .rule_pruning(rule_pruning)
        CmdRunner(builder).run(f'rule-pruning-{rule_pruning}')

    def test_rule_induction_top_down_beam_search(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .rule_induction(CmdBuilder.RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH)
        CmdRunner(builder).run('rule-induction-top-down-beam-search')

    def test_sequential_post_optimization(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .sequential_post_optimization()
        CmdRunner(builder).run('sequential-post-optimization')

    def test_feature_binning_equal_width_binary_features_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.binary) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('feature-binning-equal-width_binary-features-dense')

    def test_feature_binning_equal_width_binary_features_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.binary) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        CmdRunner(builder).run('feature-binning-equal-width_binary-features-sparse')

    def test_feature_binning_equal_width_nominal_features_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.nominal) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('feature-binning-equal-width_nominal-features-dense')

    def test_feature_binning_equal_width_nominal_features_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.nominal) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        CmdRunner(builder).run('feature-binning-equal-width_nominal-features-sparse')

    def test_feature_binning_equal_width_numerical_features_dense(self):
        builder = self._create_cmd_builder() \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('feature-binning-equal-width_numerical-features-dense')

    def test_feature_binning_equal_width_numerical_features_sparse(self):
        builder = self._create_cmd_builder() \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        CmdRunner(builder).run('feature-binning-equal-width_numerical-features-sparse')

    def test_feature_binning_equal_frequency_binary_features_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.binary) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('feature-binning-equal-frequency_binary-features-dense')

    def test_feature_binning_equal_frequency_binary_features_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.binary) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        CmdRunner(builder).run('feature-binning-equal-frequency_binary-features-sparse')

    def test_feature_binning_equal_frequency_nominal_features_dense(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.nominal) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('feature-binning-equal-frequency_nominal-features-dense')

    def test_feature_binning_equal_frequency_nominal_features_sparse(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.nominal) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        CmdRunner(builder).run('feature-binning-equal-frequency_nominal-features-sparse')

    def test_feature_binning_equal_frequency_numerical_features_dense(self):
        builder = self._create_cmd_builder() \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        CmdRunner(builder).run('feature-binning-equal-frequency_numerical-features-dense')

    def test_feature_binning_equal_frequency_numerical_features_sparse(self):
        builder = self._create_cmd_builder() \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        CmdRunner(builder).run('feature-binning-equal-frequency_numerical-features-sparse')
