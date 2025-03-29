"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of binary predictions that are part of output data.
"""
from mlrl.testbed.experiments.output.characteristics.characteristics_output import OutputCharacteristics
from mlrl.testbed.experiments.output.characteristics.matrix_output import OutputMatrix
from mlrl.testbed.experiments.problem_type import ProblemType


class PredictionCharacteristics(OutputCharacteristics):
    """
    Represents characteristics of binary predictions that are part of output data.
    """

    def __init__(self, problem_type: ProblemType, prediction_matrix: OutputMatrix):
        """
        :param problem_type:        The type of the machine learning problem, the prediction matrix corresponds to
        :param prediction_matrix:   A prediction matrix
        """
        super().__init__(problem_type,
                         prediction_matrix,
                         name='Prediction characteristics',
                         file_name='prediction_characteristics')
