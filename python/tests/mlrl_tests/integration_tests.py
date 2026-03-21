"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
import os

from abc import ABC
from typing import Any

import pytest

from .cmd_runner import CmdRunner
from .datasets import Dataset

from mlrl.testbed.experiments.state import ExperimentMode

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
        'run',
    ])
    def test_help(self, mode: str):
        builder = self._create_cmd_builder() \
            .set_mode(mode) \
            .set_show_help()
        CmdRunner(builder).run(f'help-{mode}')

    def test_single_output(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.single_output) \
            .print_evaluation()
        CmdRunner(builder).run('single-output')

    def test_model_persistence(self, dataset: Dataset):
        test_name = 'model-persistence'
        builder = self._create_cmd_builder(dataset=dataset.default) \
            .save_models() \
            .load_models()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self._create_cmd_builder() \
            .set_mode(ExperimentMode.READ) \
            .print_evaluation(False) \
            .save_evaluation(False) \
            .save_models()
        CmdRunner(builder).run(test_name, wipe_before=False)
