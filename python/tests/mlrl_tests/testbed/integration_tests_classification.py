"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

# pylint: disable=missing-function-docstring

from mlrl_tests.cmd_runner import CmdRunner
from mlrl_tests.datasets import Dataset
from mlrl_tests.integration_tests_classification import ClassificationIntegrationTests

from mlrl.testbed.experiments.state import ExperimentMode


class SklearnTestbedClassificationIntegrationTests(ClassificationIntegrationTests):
    """
    Defines a series of integration tests for the mlrl-testbed-sklearn package when using classification algorithms.
    """

    def test_label_vectors(self, dataset: Dataset):
        test_name = 'label-vectors'
        builder = self.create_cmd_builder(dataset=dataset.default) \
            .save_meta_data() \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_label_vectors() \
            .save_label_vectors()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self.create_cmd_builder() \
            .set_mode(ExperimentMode.READ) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .print_label_vectors() \
            .save_label_vectors()
        CmdRunner(builder).run(test_name, wipe_before=False)
