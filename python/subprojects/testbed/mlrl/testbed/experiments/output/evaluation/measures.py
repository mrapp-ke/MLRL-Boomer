"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing evaluation measures.
"""
from typing import Any, Callable, Iterable, List

from mlrl.testbed.experiments.output.data import OutputValue


class Measure(OutputValue):
    """
    An evaluation measure.
    """

    UNIT_SECONDS = 'seconds'

    EvaluationFunction = Callable[[Any, Any], float]

    def __init__(self,
                 option_key: str,
                 name: str,
                 evaluation_function: 'EvaluationFunction',
                 smaller_is_better: bool = False,
                 percentage: bool = True,
                 **kwargs):
        """
        :param option_key:          The key of the option that can be used for filtering
        :param name:                The name of the value
        :param evaluation_function: The function that should be invoked for evaluation
        :param smaller_is_better:   True, if smaller values are better than larger ones, False otherwise
        :param percentage:          True, if the values can be formatted as a percentage, False otherwise
        :param kwargs:              Optional keyword arguments to be passed to the evaluation function
        """
        super().__init__(option_key=option_key,
                         name=name + ' (' + ('↓' if smaller_is_better else '↑') + ')',
                         percentage=percentage)
        self.evaluation_function = evaluation_function
        self.smaller_is_better = smaller_is_better
        self.kwargs = kwargs

    @staticmethod
    def is_smaller_better(measure_name: str) -> bool:
        """
        Returns whether smaller values are better than larger values according to a measure, depending on its name.

        :param measure_name:    Name of the measure
        :return:                True, if smaller values are better than larger ones, False otherwise
        """
        return any(measure_name.find('(' + string + ')') >= 0 for string in ['↓', Measure.UNIT_SECONDS])

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


class AggregationMeasure(OutputValue):
    """
    An aggregation measure that aggregates evaluation results for several experiments.
    """

    AggregationFunction = Callable[[List[float], bool], Iterable[float]]

    def __init__(self, option_key: str, name: str, aggregation_function: 'AggregationFunction', **kwargs):
        """
        :param option_key:              The key of the option that can be used for filtering
        :param name:                    The name of the value
        :param aggregation_function:    The function that should be invoked to aggregate several evaluation results
        :param kwargs:                  Optional keyword arguments to be passed to the evaluation function
        """
        super().__init__(option_key=option_key, name=name)
        self.aggregation_function = aggregation_function
        self.kwargs = kwargs

    def aggregate(self, values: List[float], smaller_is_better: bool) -> Iterable[float]:
        """
        Applies the aggregation function to given evaluation results.

        :param values:              A list that stores the values to be aggregated
        :param smaller_is_better:   True, if smaller values are better than larger ones, False otherwise
        :return:                    An iterable that provides access to the aggregated values
        """
        return self.aggregation_function(values, smaller_is_better, **self.kwargs)
