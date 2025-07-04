"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from abc import ABC

import pytest

from .cmd_builder_classification import ClassificationCmdBuilder
from .cmd_runner import CmdRunner
from .datasets import Dataset
from .integration_tests import IntegrationTests


class ClassificationIntegrationTests(IntegrationTests, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm that can be applied to classification
    problems.
    """

    @pytest.fixture
    def dataset(self) -> Dataset:
        return Dataset()

    def test_label_vectors_train_test(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_label_vectors() \
            .store_label_vectors()
        CmdRunner(self, builder).run('label-vectors_train-test')

    def test_label_vectors_cross_validation(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_label_vectors() \
            .store_label_vectors()
        CmdRunner(self, builder).run('label-vectors_cross-validation')

    def test_label_vectors_single_fold(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_label_vectors() \
            .store_label_vectors()
        CmdRunner(self, builder).run('label-vectors_single-fold')

    def test_instance_sampling_stratified_output_wise(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .instance_sampling(ClassificationCmdBuilder.INSTANCE_SAMPLING_STRATIFIED_OUTPUT_WISE)
        CmdRunner(self, builder).run('instance-sampling-stratified-output-wise')

    def test_instance_sampling_stratified_example_wise(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .instance_sampling(ClassificationCmdBuilder.INSTANCE_SAMPLING_STRATIFIED_EXAMPLE_WISE)
        CmdRunner(self, builder).run('instance-sampling-stratified-example-wise')
