"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing model that are part of input data.
"""

from typing import Any, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import Properties
from mlrl.testbed.experiments.input.data import InputData
from mlrl.testbed.experiments.state import ExperimentState, TrainingState
from mlrl.testbed.util.log import Log


class InputModel(InputData):
    """
    Represents a model is are part of input data.
    """

    PROPERTIES = Properties(name='Model', file_name='model')

    CONTEXT = Context(include_dataset_type=False, include_prediction_scope=False)

    def __init__(self):
        super().__init__(properties=self.PROPERTIES, context=self.CONTEXT)

    @override
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        See :func:`mlrl.testbed.experiments.input.data.InputData.update_state`
        """
        Log.info('Successfully loaded model')
        state.training_result = TrainingState(learner=input_data)
