"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of binary predictions that are part of output data.
"""
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import OutputCharacteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.matrix_output import OutputMatrix

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import TabularProperties
from mlrl.testbed.experiments.problem_domain import ProblemDomain


class PredictionCharacteristics(OutputCharacteristics):
    """
    Represents characteristics of binary predictions that are part of output data.
    """

    PROPERTIES = TabularProperties(name='Prediction characteristics', file_name='prediction_characteristics')

    CONTEXT = Context()

    def __init__(self, problem_domain: ProblemDomain, prediction_matrix: OutputMatrix):
        """
        :param problem_domain:      The problem domain, the prediction matrix corresponds to
        :param prediction_matrix:   A prediction matrix
        """
        super().__init__(problem_domain=problem_domain,
                         output_matrix=prediction_matrix,
                         properties=self.PROPERTIES,
                         context=self.CONTEXT)
