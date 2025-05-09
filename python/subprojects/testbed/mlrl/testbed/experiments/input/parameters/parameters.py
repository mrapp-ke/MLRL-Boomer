"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing algorithmic parameters that are part of input data.
"""
from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import Table


class InputParameters(TabularInputData):
    """
    Represents algorithmic parameters, set by the user, that are part of input data.
    """

    def __init__(self):
        super().__init__(TabularInputData.Properties(file_name='parameters', has_header=True),
                         Data.Context(include_dataset_type=False, include_prediction_scope=False))

    def _update_state(self, state: ExperimentState, table: Table):
        parameter_dict = {}

        for column in table.to_column_wise_table().columns:
            parameter_name = column.header

            for parameter_value in column:
                parameter_dict[parameter_name] = parameter_value

        state.parameters = parameter_dict
