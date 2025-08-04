"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing model that are part of input data.
"""
import logging as log

from typing import Any, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.input.data import InputData
from mlrl.testbed.experiments.state import ExperimentState, TrainingState


class InputModel(InputData):
    """
    Represents a model is are part of input data.
    """

    def __init__(self):
        super().__init__(InputData.Properties(file_name='model'),
                         Context(include_dataset_type=False, include_prediction_scope=False))

    @override
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        See :func:`mlrl.testbed.experiments.input.data.InputData.update_state`
        """
        log.info('Successfully loaded model')
        state.training_result = TrainingState(learner=input_data)
