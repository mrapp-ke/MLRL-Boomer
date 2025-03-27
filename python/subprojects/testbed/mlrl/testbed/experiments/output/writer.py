"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing output data to sinks.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional

from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.state import ExperimentState, PredictionResult


class OutputWriter(ABC):
    """
    An abstract base class for all classes that allow to write output data to one or several sinks.
    """

    def __init__(self, *sinks: Sink):
        """
        :param sinks: The sinks, output data should be written to
        """
        self.sinks = list(sinks)

    def write_output(self, state: ExperimentState, prediction_result: Optional[PredictionResult] = None):
        """
        Generates the output data and writes it to all available sinks.

        :param state:               The state from which the output data should be generated
        :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if no
                                    predictions have been obtained
        """
        sinks = self.sinks

        if sinks:
            output_data = self._generate_output_data(state, prediction_result)

            if output_data:
                for sink in sinks:
                    sink.write_to_sink(state, prediction_result, output_data)

    @abstractmethod
    def _generate_output_data(self, state: ExperimentState,
                              prediction_result: Optional[PredictionResult]) -> Optional[Any]:
        """
        Must be implemented by subclasses in order to generate the output data that should be written to the available
        sinks.

        :param state:               The state from which the output data should be generated
        :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if no
                                    predictions have been obtained
        :return:                    The output data that has been generated or None, if no output data was generated
        """
