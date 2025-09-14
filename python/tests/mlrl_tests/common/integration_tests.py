"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
import os

from abc import ABC
from typing import Any, Optional

import pytest

from .cmd_runner import CmdRunner
from .datasets import Dataset

from mlrl.common.config.parameters import BINNING_EQUAL_FREQUENCY, BINNING_EQUAL_WIDTH, SAMPLING_WITH_REPLACEMENT, \
    SAMPLING_WITHOUT_REPLACEMENT, OutputSamplingParameter, PostOptimizationParameter, RuleInductionParameter, \
    RulePruningParameter
from mlrl.common.learners import SparsePolicy

from mlrl.testbed_sklearn.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments

from mlrl.testbed.modes import Mode

from mlrl.util.cli import NONE
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
    @pytest.mark.parametrize('mode', [
        'single',
        'batch',
        'read',
    ])
    def test_help(self, mode: str):
        builder = self._create_cmd_builder() \
            .set_mode(mode) \
            .set_show_help()
        CmdRunner(builder).run(f'help-{mode}')

    def test_print_all(self):
        builder = self._create_cmd_builder() \
            .print_all(True)
        CmdRunner(builder).run('print-all')

    def test_save_all(self):
        builder = self._create_cmd_builder() \
            .save_all(True)
        CmdRunner(builder).run('save-all')

    def test_batch_mode(self):
        builder = self._create_cmd_builder() \
            .set_mode(Mode.MODE_BATCH) \
            .save_all()
        CmdRunner(builder).run('batch-mode')

    def test_batch_mode_separate_folds(self):
        builder = self._create_cmd_builder() \
            .set_mode(Mode.MODE_BATCH) \
            .save_all() \
            .data_split(DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
                        options=Options({DatasetSplitterArguments.OPTION_NUM_FOLDS: 2}))
        CmdRunner(builder).run('batch-mode-separate-folds')

    def test_batch_mode_list(self):
        builder = self._create_cmd_builder() \
            .set_mode(Mode.MODE_BATCH, '--list')
        CmdRunner(builder).run('batch-mode-list')

    def test_batch_mode_slurm(self):
        builder = self._create_cmd_builder() \
            .set_mode(Mode.MODE_BATCH) \
            .set_runner('slurm')
        CmdRunner(builder).run('batch-mode-slurm')

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

    def test_meta_data(self):
        builder = self._create_cmd_builder() \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_meta_data() \
            .save_meta_data()
        CmdRunner(builder).run('meta_data')

    @pytest.mark.parametrize('data_split, data_split_options, predefined', [
        (NONE, Options(), False),
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options(), False),
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options(), True),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options(), False),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options(), True),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
         Options({
             DatasetSplitterArguments.OPTION_FIRST_FOLD: 1,
             DatasetSplitterArguments.OPTION_LAST_FOLD: 1,
         }), False),
    ])
    def test_evaluation(self, data_split: str, data_split_options: Options, predefined: bool, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default + ('-predefined' if predefined else '')) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation() \
            .save_evaluation()
        CmdRunner(builder).run(f'evaluation_{data_split}' + ('-predefined' if predefined else '')
                               + (f'_{data_split_options}' if data_split_options else ''))

    def test_evaluation_training_data(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .predict_for_training_data() \
            .print_evaluation() \
            .save_evaluation()
        CmdRunner(builder).run('evaluation_training-data')

    def test_evaluation_incremental(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .incremental_evaluation() \
            .print_evaluation() \
            .save_evaluation()
        CmdRunner(builder).run('evaluation_incremental')

    @pytest.mark.parametrize('data_split, data_split_options', [
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
         Options({
             DatasetSplitterArguments.OPTION_FIRST_FOLD: 1,
             DatasetSplitterArguments.OPTION_LAST_FOLD: 1,
         })),
    ])
    def test_model_persistence(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .save_models() \
            .load_models()
        CmdRunner(builder).run(f'model-persistence_{data_split}'
                               + (f'_{data_split_options}' if data_split_options else ''))

    @pytest.mark.parametrize('data_split, data_split_options', [
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
         Options({
             DatasetSplitterArguments.OPTION_FIRST_FOLD: 1,
             DatasetSplitterArguments.OPTION_LAST_FOLD: 1,
         })),
    ])
    def test_predictions(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_predictions() \
            .print_ground_truth() \
            .save_predictions() \
            .save_ground_truth()
        CmdRunner(builder).run(f'predictions_{data_split}' + (f'_{data_split_options}' if data_split_options else ''))

    def test_predictions_training_data(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .predict_for_training_data() \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_predictions() \
            .print_ground_truth() \
            .save_predictions() \
            .save_ground_truth()
        CmdRunner(builder).run('predictions_training-data')

    @pytest.mark.parametrize('data_split, data_split_options', [
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
         Options({
             DatasetSplitterArguments.OPTION_FIRST_FOLD: 1,
             DatasetSplitterArguments.OPTION_LAST_FOLD: 1,
         })),
    ])
    def test_prediction_characteristics(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_prediction_characteristics() \
            .save_prediction_characteristics()
        CmdRunner(builder).run(f'prediction-characteristics_{data_split}'
                               + (f'_{data_split_options}' if data_split_options else ''))

    def test_prediction_characteristics_training_data(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .predict_for_training_data() \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_prediction_characteristics() \
            .save_prediction_characteristics()
        CmdRunner(builder).run('prediction-characteristics_training-data')

    @pytest.mark.parametrize('data_split, data_split_options', [
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
         Options({
             DatasetSplitterArguments.OPTION_FIRST_FOLD: 1,
             DatasetSplitterArguments.OPTION_LAST_FOLD: 1,
         })),
    ])
    def test_data_characteristics(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_data_characteristics() \
            .save_data_characteristics()
        CmdRunner(builder).run(f'data-characteristics_{data_split}'
                               + (f'_{data_split_options}' if data_split_options else ''))

    @pytest.mark.parametrize('data_split, data_split_options', [
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
         Options({
             DatasetSplitterArguments.OPTION_FIRST_FOLD: 1,
             DatasetSplitterArguments.OPTION_LAST_FOLD: 1,
         })),
    ])
    def test_model_characteristics(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_model_characteristics() \
            .save_model_characteristics()
        CmdRunner(builder).run(f'model-characteristics_{data_split}'
                               + (f'_{data_split_options}' if data_split_options else ''))

    @pytest.mark.parametrize('data_split, data_split_options', [
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
         Options({
             DatasetSplitterArguments.OPTION_FIRST_FOLD: 1,
             DatasetSplitterArguments.OPTION_LAST_FOLD: 1,
         })),
    ])
    def test_rules(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_rules() \
            .save_rules()
        CmdRunner(builder).run(f'rules_{data_split}' + (f'_{data_split_options}' if data_split_options else ''))

    @pytest.mark.parametrize('dataset_name', ['numerical_sparse', 'binary', 'nominal', 'ordinal'])
    @pytest.mark.parametrize('feature_format', [SparsePolicy.FORCE_DENSE, SparsePolicy.FORCE_SPARSE])
    def test_feature_format(self, dataset_name: str, feature_format: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=getattr(dataset, dataset_name)) \
            .feature_format(feature_format)
        CmdRunner(builder).run(f'feature-format-{dataset_name}-{feature_format}')

    @pytest.mark.parametrize('output_format', [
        SparsePolicy.FORCE_DENSE,
        SparsePolicy.FORCE_SPARSE,
    ])
    def test_output_format(self, output_format: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .output_format(output_format)
        CmdRunner(builder).run(f'output-format-{output_format}')

    @pytest.mark.parametrize('prediction_format', [
        SparsePolicy.FORCE_DENSE,
        SparsePolicy.FORCE_SPARSE,
    ])
    def test_prediction_format(self, prediction_format: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .prediction_format(prediction_format) \
            .print_predictions() \
            .print_ground_truth()
        CmdRunner(builder).run(f'prediction-format-{prediction_format}')

    @pytest.mark.parametrize('data_split, data_split_options', [
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
         Options({
             DatasetSplitterArguments.OPTION_FIRST_FOLD: 1,
             DatasetSplitterArguments.OPTION_LAST_FOLD: 1
         })),
    ])
    def test_parameters(self, data_split: str, data_split_options: Options, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .save_parameters() \
            .load_parameters()
        CmdRunner(builder).run(f'parameters_{data_split}' + (f'_{data_split_options}' if data_split_options else ''))

    @pytest.mark.parametrize('instance_sampling', [
        NONE,
        SAMPLING_WITH_REPLACEMENT,
        SAMPLING_WITHOUT_REPLACEMENT,
    ])
    def test_instance_sampling(self, instance_sampling: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .instance_sampling(instance_sampling)
        CmdRunner(builder).run(f'instance-sampling-{instance_sampling}')

    @pytest.mark.parametrize('feature_sampling', [
        NONE,
        SAMPLING_WITHOUT_REPLACEMENT,
    ])
    def test_feature_sampling(self, feature_sampling: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .feature_sampling(feature_sampling)
        CmdRunner(builder).run(f'feature-sampling-{feature_sampling}')

    @pytest.mark.parametrize('output_sampling', [
        NONE,
        OutputSamplingParameter.OUTPUT_SAMPLING_ROUND_ROBIN,
        SAMPLING_WITHOUT_REPLACEMENT,
    ])
    def test_output_sampling(self, output_sampling: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .output_sampling(output_sampling)
        CmdRunner(builder).run(f'output-sampling-{output_sampling}')

    @pytest.mark.parametrize('rule_pruning, instance_sampling', [
        (NONE, None),
        (RulePruningParameter.RULE_PRUNING_IREP, SAMPLING_WITHOUT_REPLACEMENT),
    ])
    def test_rule_pruning(self, rule_pruning: str, instance_sampling: Optional[str], dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .instance_sampling(instance_sampling) \
            .rule_pruning(rule_pruning)
        CmdRunner(builder).run(f'rule-pruning-{rule_pruning}')

    @pytest.mark.parametrize('rule_induction', [RuleInductionParameter.RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH])
    def test_rule_induction(self, rule_induction: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .rule_induction(rule_induction)
        CmdRunner(builder).run(f'rule-induction-{rule_induction}')

    @pytest.mark.parametrize('post_optimization', [PostOptimizationParameter.POST_OPTIMIZATION_SEQUENTIAL])
    def test_post_optimization(self, post_optimization: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .post_optimization(post_optimization)
        CmdRunner(builder).run(f'post-optimization-{post_optimization}')

    @pytest.mark.parametrize('feature_binning', [
        BINNING_EQUAL_WIDTH,
        BINNING_EQUAL_FREQUENCY,
    ])
    @pytest.mark.parametrize('dataset_name', ['numerical', 'nominal', 'binary'])
    @pytest.mark.parametrize('feature_format', [SparsePolicy.FORCE_DENSE, SparsePolicy.FORCE_SPARSE])
    def test_feature_binning(self, feature_binning: str, dataset_name: str, feature_format: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=getattr(dataset, dataset_name)) \
            .feature_binning(feature_binning) \
            .feature_format(feature_format)
        CmdRunner(builder).run(f'feature-binning-{feature_binning}_{dataset_name}-features-{feature_format}')
