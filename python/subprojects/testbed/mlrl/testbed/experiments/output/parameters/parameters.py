"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing algorithmic parameters that are part of output data.
"""

from typing import override

from mlrl.testbed.experiments.input.parameters.parameters import InputParameters
from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.state import ParameterDict
from mlrl.testbed.experiments.table import RowWiseTable, Table
from rich.console import ConsoleRenderable

from mlrl.util.options import Options


class OutputParameters(TabularOutputData):
    """
    Represents algorithmic parameters, set by the user, that are part of output data.
    """

    def __init__(self, parameter_dict: ParameterDict):
        """
        :param parameter_dict: A dictionary that stores the parameters of a learner
        """
        super().__init__(properties=InputParameters.PROPERTIES, context=InputParameters.CONTEXT)
        self.custom_parameters = {key: value for key, value in parameter_dict.items() if value is not None}

    @override
    def to_text(self, options: Options, **kwargs) -> str | ConsoleRenderable | None:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        table = self.to_table(options, **kwargs)
        return table.to_column_wise_table().sort_by_headers().to_rich_table() if table else None

    @override
    def to_table(self, options: Options, **_) -> Table | None:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        parameters = self.custom_parameters
        parameter_names = parameters.keys()
        parameter_values = map(lambda parameter_name: parameters[parameter_name], parameter_names)
        return RowWiseTable(*parameter_names).add_row(*parameter_values)
