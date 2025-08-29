"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing algorithmic parameters that are part of output data.
"""
from typing import Optional, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.output.data import OutputData, TabularOutputData
from mlrl.testbed.experiments.state import ParameterDict
from mlrl.testbed.experiments.table import RowWiseTable, Table

from mlrl.util.options import Options


class OutputParameters(TabularOutputData):
    """
    Represents algorithmic parameters, set by the user, that are part of output data.
    """

    def __init__(self, parameter_dict: ParameterDict):
        """
        :param parameter_dict: A dictionary that stores the parameters of a learner
        """
        super().__init__(OutputData.Properties(name='Custom parameters', file_name='parameters'),
                         Context(include_dataset_type=False))
        self.custom_parameters = {key: value for key, value in parameter_dict.items() if value is not None}

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        table = self.to_table(options, **kwargs)

        if table:
            return table.to_column_wise_table().sort_by_headers().format()
        return None

    @override
    def to_table(self, options: Options, **_) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        parameters = self.custom_parameters
        parameter_names = parameters.keys()
        parameter_values = map(lambda parameter_name: parameters[parameter_name], parameter_names)
        return RowWiseTable(*parameter_names).add_row(*parameter_values)
