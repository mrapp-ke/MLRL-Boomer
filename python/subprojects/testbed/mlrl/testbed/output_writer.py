"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for writing output data to sinks like the console or output files.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mlrl.common.config.options import Options

from mlrl.testbed.data_sinks import Sink
from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.training_result import TrainingResult


class Tabularizable(ABC):
    """
    An abstract base class for all classes from which a tabular representation can be created.
    """

    @abstractmethod
    def tabularize(self, options: Options, **kwargs) -> Optional[List[Dict[str, str]]]:
        """
        Creates and returns a tabular representation of the object.

        :param options: Options to be taken into account
        :return:        The tabular representation that has been created
        """


class OutputWriter(ABC):
    """
    An abstract base class for all classes that allow to write output data to one or several sinks, e.g., the console or
    output files.
    """

    KWARG_DATA_SPLIT = 'data_split'

    def __init__(self, sinks: List[Sink]):
        """
        :param sinks: A list that contains all sinks, output data should be written to
        """
        self.sinks = sinks

    @abstractmethod
    def _generate_output_data(self, scope: OutputScope, training_result: Optional[TrainingResult],
                              prediction_result: Optional[PredictionResult]) -> Optional[Any]:
        """
        Must be implemented by subclasses in order to generate the output data that should be written to the available
        sinks.

        :param scope:               The scope of the output data
        :param training_result:     A `TrainingResult` that stores the result of a training process or None, if no model
                                    has been trained
        :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if no
                                    predictions have been obtained
        :return:                    The output data that has been generated or None, if no output data was generated
        """

    def write_output(self,
                     scope: OutputScope,
                     training_result: Optional[TrainingResult] = None,
                     prediction_result: Optional[PredictionResult] = None):
        """
        Generates the output data and writes it to all available sinks.

        :param scope:               The scope of the output data
        :param training_result:     A `TrainingResult` that stores the result of a training process or None, if no model
                                    has been trained
        :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if no
                                    predictions have been obtained
        """
        sinks = self.sinks

        if sinks:
            output_data = self._generate_output_data(scope, training_result, prediction_result)

            if output_data:
                for sink in sinks:
                    sink.write_output(scope, training_result, prediction_result, output_data)
