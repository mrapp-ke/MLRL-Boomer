"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing sinks, output data may be written to.
"""
from abc import ABC, abstractmethod
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.training_result import TrainingResult


class Sink(ABC):
    """
    An abstract base class for all sinks, output data may be written to.
    """

    def __init__(self, options: Options = Options()):
        """
        :param options: Options to be taken into account
        """
        self.options = options

    @abstractmethod
    def write_to_sink(self, scope: OutputScope, training_result: Optional[TrainingResult],
                      prediction_result: Optional[PredictionResult], output_data, **kwargs):
        """
        Must be implemented by subclasses in order to write output data to the sink.

        :param scope:               The scope of the output data
        :param training_result:     A `TrainingResult` that stores the result of a training process or None, if no
                                    model has been trained
        :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if
                                    no predictions have been obtained
        :param output_data:         The output data that should be written to the sink
        """
