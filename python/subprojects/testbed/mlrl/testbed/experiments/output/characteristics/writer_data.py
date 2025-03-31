"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing characteristics of datasets to one or several sinks.
"""
from typing import Optional

from mlrl.testbed.experiments.output.characteristics.characteristics_data import DataCharacteristics
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class DataCharacteristicsWriter(OutputWriter):
    """
    Allows writing characteristics of a dataset to one or several sinks.
    """

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        dataset = state.dataset
        return DataCharacteristics(problem_type=state.problem_type, dataset=dataset)
