"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing evaluation measures.
"""
from typing import Any, Callable

from mlrl.testbed.experiments.output.data import OutputValue


class Measure(OutputValue):
    """
    An evaluation measure.
    """

    EvaluationFunction = Callable[[Any, Any], float]

    def __init__(self,
                 option_key: str,
                 name: str,
                 evaluation_function: EvaluationFunction,
                 percentage: bool = True,
                 **kwargs):
        """
        :param option_key:          The key of the option that can be used for filtering
        :param name:                The name of the value
        :param evaluation_function: The function that should be invoked for evaluation
        :param percentage:          True, if the values can be formatted as a percentage, False otherwise
        :param kwargs:              Optional keyword arguments to be passed to the evaluation function
        """
        super().__init__(option_key=option_key, name=name, percentage=percentage)
        self.evaluation_function = evaluation_function
        self.kwargs = kwargs

    def evaluate(self, ground_truth: Any, predictions: Any) -> float:
        """
        Applies the evaluation function to given predictions and the corresponding ground truth.

        :param ground_truth:    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the ground truth
        :param predictions:     A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the predictions to be evaluated
        :return:                An evaluation score in [0, 1]
        """
        return self.evaluation_function(ground_truth, predictions, **self.kwargs)
