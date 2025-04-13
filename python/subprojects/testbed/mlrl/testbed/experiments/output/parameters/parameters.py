"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing algorithmic parameters that are part of output data.
"""
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.state import ParameterDict
from mlrl.testbed.experiments.table import RowWiseTable, Table


class CustomParameters(TabularOutputData):
    """
    Represents algorithmic parameters, set by the user, that are part of output data.
    """

    def __init__(self, parameter_dict: ParameterDict):
        """
        :param parameter_dict: A dictionary that stores the parameters of a learner
        """
        super().__init__(name='Custom parameters',
                         file_name='parameters',
                         default_context=Data.Context(include_dataset_type=False))
        self.custom_parameters = {key: value for key, value in parameter_dict.items() if value is not None}

    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        return self.to_table(options, **kwargs).to_column_wise_table().sort_by_headers().format()

    # pylint: disable=unused-argument
    def to_table(self, options: Options, **_) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        parameters = self.custom_parameters
        parameter_names = parameters.keys()
        parameter_values = map(lambda parameter_name: parameters[parameter_name], parameter_names)
        return RowWiseTable(*parameter_names).add_row(*parameter_values)
