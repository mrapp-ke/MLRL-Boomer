"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of binary predictions that are part of output data.
"""
from typing import Any, List, Tuple

from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import LABEL_CHARACTERISTICS, \
    OUTPUT_CHARACTERISTICS, Characteristic, OutputCharacteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.matrix_output import OutputMatrix

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import TabularProperties
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain


class PredictionCharacteristics(OutputCharacteristics):
    """
    Represents characteristics of binary predictions that are part of output data.
    """

    PROPERTIES = TabularProperties(name='Prediction characteristics', file_name='prediction_characteristics')

    CONTEXT = Context()

    def __init__(self, values: List[Tuple[Characteristic, Any]]):
        """
        :param values: A list that stores different data characteristics and their corresponding values
        """
        super().__init__(values=values, properties=self.PROPERTIES, context=self.CONTEXT)

    @staticmethod
    def from_prediction_matrix(problem_domain: ProblemDomain,
                               prediction_matrix: OutputMatrix) -> 'PredictionCharacteristics':
        """
        Creates and returns `OutputCharacteristics` from a given output matrix.

        :param problem_domain:      The problem domain, the output matrix corresponds to
        :param prediction_matrix:   An prediction matrix
        :return:                    The `OutputCharacteristics` that have been created
        """
        if isinstance(problem_domain, ClassificationProblem):
            characteristics = LABEL_CHARACTERISTICS
        else:
            characteristics = OUTPUT_CHARACTERISTICS

        return PredictionCharacteristics([(characteristic, characteristic.function(prediction_matrix))
                                          for characteristic in characteristics])
