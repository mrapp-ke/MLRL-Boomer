"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing algorithmic parameters to one or several sinks.
"""

from typing import Optional

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.parameters.parameters import CustomParameters
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class ParameterWriter(OutputWriter):
    """
    Allows writing algorithmic parameters to one or several sinks.
    """

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        return CustomParameters(state.parameters)
