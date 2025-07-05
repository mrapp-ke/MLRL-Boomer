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

from mlrl.testbed_sklearn.experiments.input.dataset.splitters.extension import OPTION_FIRST_FOLD, OPTION_LAST_FOLD

from mlrl.util.options import Options

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
            .data_split(CmdBuilder.DATA_SPLIT_NO) \
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
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION) \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_cross-validation')

    def test_evaluation_cross_validation_predefined(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default + '-predefined') \
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION) \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(builder).run('evaluation_cross-validation-predefined')

    def test_evaluation_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options({OPTION_FIRST_FOLD: 1, OPTION_LAST_FOLD: 1})) \
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
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION) \
            .set_model_dir()
        CmdRunner(builder).run('model-persistence_cross-validation')

    def test_model_persistence_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options({OPTION_FIRST_FOLD: 1, OPTION_LAST_FOLD: 1})) \
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
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_predictions() \
            .print_ground_truth() \
            .store_predictions() \
            .store_ground_truth()
        CmdRunner(builder).run('predictions_cross-validation')

    def test_predictions_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options({OPTION_FIRST_FOLD: 1, OPTION_LAST_FOLD: 1})) \
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
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        CmdRunner(builder).run('prediction-characteristics_cross-validation')

    def test_prediction_characteristics_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options({OPTION_FIRST_FOLD: 1, OPTION_LAST_FOLD: 1})) \
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
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_data_characteristics() \
            .store_data_characteristics()
        CmdRunner(builder).run('data-characteristics_cross-validation')

    def test_data_characteristics_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options({OPTION_FIRST_FOLD: 1, OPTION_LAST_FOLD: 1})) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_data_characteristics() \
            .store_data_characteristics()
        CmdRunner(builder).run('data-characteristics_single-fold')

    @pytest.mark.parametrize('data_split, data_split_options', [
        (CmdBuilder.DATA_SPLIT_TRAIN_TEST, Options()),
        (CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options()),
        (CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options({
            OPTION_FIRST_FOLD: 1,
            OPTION_LAST_FOLD: 1,
        })),
    ])
    def test_model_characteristics(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .store_model_characteristics()
        CmdRunner(builder).run(f'model-characteristics_{data_split}'
                               + (f'_{data_split_options}' if data_split_options else ''))

    @pytest.mark.parametrize('data_split, data_split_options', [
        (CmdBuilder.DATA_SPLIT_TRAIN_TEST, Options()),
        (CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options()),
        (CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options({
            OPTION_FIRST_FOLD: 1,
            OPTION_LAST_FOLD: 1,
        })),
    ])
    def test_rules(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_rules() \
            .store_rules()
        CmdRunner(builder).run(f'rules_{data_split}' + (f'_{data_split_options}' if data_split_options else ''))

    @pytest.mark.parametrize('dataset_name', ['numerical_sparse', 'binary', 'nominal', 'ordinal'])
    @pytest.mark.parametrize('feature_format', [CmdBuilder.FEATURE_FORMAT_DENSE, CmdBuilder.FEATURE_FORMAT_SPARSE])
    def test_feature_format(self, dataset_name: str, feature_format: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=getattr(dataset, dataset_name)) \
            .feature_format(feature_format)
        CmdRunner(builder).run(f'feature-format-{dataset_name}-{feature_format}')

    @pytest.mark.parametrize('output_format', [
        CmdBuilder.OUTPUT_FORMAT_DENSE,
        CmdBuilder.OUTPUT_FORMAT_SPARSE,
    ])
    def test_output_format(self, output_format: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .output_format(output_format)
        CmdRunner(builder).run(f'output-format-{output_format}')

    @pytest.mark.parametrize('prediction_format', [
        CmdBuilder.PREDICTION_FORMAT_DENSE,
        CmdBuilder.PREDICTION_FORMAT_SPARSE,
    ])
    def test_prediction_format(self, prediction_format: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .prediction_format(prediction_format) \
            .print_predictions() \
            .print_ground_truth()
        CmdRunner(builder).run(f'prediction-format-{prediction_format}')

    @pytest.mark.parametrize('data_split, data_split_options', [
        (CmdBuilder.DATA_SPLIT_TRAIN_TEST, Options()),
        (CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options()),
        (CmdBuilder.DATA_SPLIT_CROSS_VALIDATION, Options({
            OPTION_FIRST_FOLD: 1,
            OPTION_LAST_FOLD: 1
        })),
    ])
    def test_parameters(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .set_parameter_save_dir() \
            .set_parameter_load_dir()
        CmdRunner(builder).run(f'parameters_{data_split}' + (f'_{data_split_options}' if data_split_options else ''))

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

    @pytest.mark.parametrize('rule_induction', [CmdBuilder.RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH])
    def test_rule_induction(self, rule_induction: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .rule_induction(rule_induction)
        CmdRunner(builder).run(f'rule-induction-{rule_induction}')

    def test_sequential_post_optimization(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .sequential_post_optimization()
        CmdRunner(builder).run('sequential-post-optimization')

    @pytest.mark.parametrize('feature_binning', [
        CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH,
        CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY,
    ])
    @pytest.mark.parametrize('dataset_name', ['numerical', 'nominal', 'binary'])
    @pytest.mark.parametrize('feature_format', [CmdBuilder.FEATURE_FORMAT_DENSE, CmdBuilder.FEATURE_FORMAT_SPARSE])
    def test_feature_binning(self, feature_binning: str, dataset_name: str, feature_format: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=getattr(dataset, dataset_name)) \
            .feature_binning(feature_binning) \
            .feature_format(feature_format)
        CmdRunner(builder).run(f'feature-binning-{feature_binning}_{dataset_name}-features-{feature_format}')
