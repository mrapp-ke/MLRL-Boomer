"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring

from typing import Optional

import pytest

from .cmd_runner import CmdRunner
from .datasets import Dataset
from .integration_tests import IntegrationTests

from mlrl.common.config.parameters import BINNING_EQUAL_FREQUENCY, BINNING_EQUAL_WIDTH, \
    SAMPLING_STRATIFIED_EXAMPLE_WISE, SAMPLING_STRATIFIED_OUTPUT_WISE, SAMPLING_WITH_REPLACEMENT, \
    SAMPLING_WITHOUT_REPLACEMENT, OutputSamplingParameter, PostOptimizationParameter, RuleInductionParameter, \
    RulePruningParameter
from mlrl.common.learners import SparsePolicy

from mlrl.testbed.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments
from mlrl.testbed.experiments.state import ExperimentMode

from mlrl.util.cli import NONE
from mlrl.util.options import Options


class RuleLearnerIntegrationTestsMixin(IntegrationTests):
    """
    A mixin for integration tests for a rule learning algorithm.
    """

    def test_sparse_feature_value(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.numerical_sparse) \
            .sparse_feature_value(1.0)
        CmdRunner(builder).run('sparse-feature-value')

    def test_evaluation_incremental(self, dataset: Dataset):
        test_name = 'evaluation_incremental'
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .incremental_evaluation() \
            .print_evaluation() \
            .save_evaluation()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self._create_cmd_builder() \
            .set_mode(ExperimentMode.READ) \
            .print_evaluation() \
            .save_evaluation()
        CmdRunner(builder).run(test_name, wipe_before=False)

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
        test_name = f'model-characteristics_{data_split}' + (f'_{data_split_options}' if data_split_options else '')
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_model_characteristics() \
            .save_model_characteristics()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self._create_cmd_builder() \
            .set_mode(ExperimentMode.READ) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_model_characteristics() \
            .save_model_characteristics()
        CmdRunner(builder).run(test_name, wipe_before=False)

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
        test_name = f'rules_{data_split}' + (f'_{data_split_options}' if data_split_options else '')
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_rules() \
            .save_rules()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self._create_cmd_builder() \
            .set_mode(ExperimentMode.READ) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_rules() \
            .save_rules()
        CmdRunner(builder).run(test_name, wipe_before=False)

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


class ClassificationRuleLearnerIntegrationTestsMixin(RuleLearnerIntegrationTestsMixin):
    """
    A mixin for integration tests for a classification rule learner.
    """

    @pytest.mark.parametrize('instance_sampling', [
        SAMPLING_STRATIFIED_OUTPUT_WISE,
        SAMPLING_STRATIFIED_EXAMPLE_WISE,
    ])
    def test_instance_sampling_stratified(self, instance_sampling: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .instance_sampling(instance_sampling)
        CmdRunner(builder).run(f'instance-sampling-{instance_sampling}')
