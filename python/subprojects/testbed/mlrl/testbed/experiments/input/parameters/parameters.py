"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing algorithmic parameters that are part of input data.
"""
from typing import override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.state import ExperimentState, ParameterDict
from mlrl.testbed.experiments.table import Table


class InputParameters(TabularInputData):
    """
    Represents algorithmic parameters, set by the user, that are part of input data.
    """

    def __init__(self):
        super().__init__(TabularInputData.Properties(file_name='parameters', has_header=True),
                         Context(include_dataset_type=False, include_prediction_scope=False))

    @override
    def _update_state(self, state: ExperimentState, table: Table):
        parameter_dict: ParameterDict = {}

        for column in table.to_column_wise_table().columns:
            parameter_name = column.header

            if parameter_name:
                for parameter_value in column:
                    if parameter_value:
                        parameter_dict[parameter_name] = parameter_value

        state.parameters = parameter_dict
