"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing algorithmic parameters that are part of output data.
"""
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.output.table import RowWiseTable, Table
from mlrl.testbed.experiments.state import ExperimentState, ParameterDict
from mlrl.testbed.util.format import format_table


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
                         default_formatter_options=ExperimentState.FormatterOptions(include_dataset_type=False))
        self.custom_parameters = {key: value for key, value in parameter_dict.items() if value is not None}

    # pylint: disable=unused-argument
    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        custom_parameters = self.custom_parameters
        rows = [[str(key), str(custom_parameters[key])] for key in sorted(custom_parameters)]
        return format_table(rows)

    # pylint: disable=unused-argument
    def to_table(self, options: Options, **_) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        return RowWiseTable.from_dict(self.custom_parameters)
