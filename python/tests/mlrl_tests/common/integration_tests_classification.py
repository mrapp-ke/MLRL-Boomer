"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from abc import ABC

import pytest

from .cmd_runner import CmdRunner
from .datasets import Dataset
from .integration_tests import IntegrationTests, RuleLearnerIntegrationTestsMixin

from mlrl.common.config.parameters import SAMPLING_STRATIFIED_EXAMPLE_WISE, SAMPLING_STRATIFIED_OUTPUT_WISE

from mlrl.testbed.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments
from mlrl.testbed.experiments.state import ExperimentMode

from mlrl.util.options import Options


class ClassificationIntegrationTests(IntegrationTests, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm that can be applied to classification
    problems.
    """

    @pytest.fixture
    def dataset(self) -> Dataset:
        return Dataset()

    @pytest.mark.parametrize('data_split, data_split_options', [
        (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options()),
        (DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
         Options({
             DatasetSplitterArguments.OPTION_FIRST_FOLD: 1,
             DatasetSplitterArguments.OPTION_LAST_FOLD: 1,
         })),
    ])
    def test_label_vectors(self, data_split: str, data_split_options: Options, dataset: Dataset):
        test_name = f'label-vectors_{data_split}' + (f'_{data_split_options}' if data_split_options else '')
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .data_split(data_split, options=data_split_options) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_label_vectors() \
            .save_label_vectors()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self._create_cmd_builder() \
            .set_mode(ExperimentMode.READ) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_label_vectors() \
            .save_label_vectors()
        CmdRunner(builder).run(test_name, wipe_before=False)


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
