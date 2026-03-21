"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from abc import ABC

import pytest

from .cmd_runner import CmdRunner
from .datasets import Dataset
from .integration_tests import IntegrationTests

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

    def test_label_vectors(self, dataset: Dataset):
        test_name = 'label-vectors'
        builder = self._create_cmd_builder(dataset=dataset.default) \
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
