"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for keeping track of several measurements according to different measures.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from mlrl.testbed.experiments.output.evaluation.measures import Measure


class Measurements:
    """
    Keeps track of measurements according to different measures.
    """

    def __init__(self):
        self.measures = set()
        self.results = None

    def put(self, measure: Measure, value: float, num_folds: int, fold: Optional[int]):
        """
        Adds a new measurement according to a given measure.

        :param measure:     The measure
        :param value:       The value according to the measure
        :param num_folds:   The total number of cross validation folds
        :param fold:        The fold, the score corresponds to, or None, if no cross validation is used
        """
        results = self.results

        if not results:
            results = [{} for _ in range(num_folds)]
            self.results = results

        if len(results) != num_folds:
            raise AssertionError('Inconsistent number of total folds given')

        self.measures.add(measure)
        values = results[0 if fold is None else fold]
        values[measure] = value

    def get(self, measure: Measure, fold: Optional[int], **kwargs) -> str:
        """
        Returns a textual representation of a value according to a given measure.

        :param measure: The measure
        :param fold:    The fold, the value corresponds to, or None, if no cross validation is used
        :return:        A textual representation of the value
        """
        results = self.results

        if not results:
            raise AssertionError('No evaluation results available')

        value = results[0 if fold is None else fold][measure]
        return measure.format(value, **kwargs)

    def dict(self, fold: Optional[int], **kwargs) -> Dict[Measure, str]:
        """
        Returns a dictionary that stores the values for a specific fold according to each measure.

        :param fold:    The fold, the values correspond to, or None, if no cross validation is used
        :return:        A dictionary that stores textual representations of the values for the given fold according
                        to each measure
        """
        results = self.results

        if not results:
            raise AssertionError('No evaluation results available')

        result_dict = {}

        for measure, score in results[0 if fold is None else fold].items():
            result_dict[measure] = measure.format(score, **kwargs)

        return result_dict

    def avg(self, measure: Measure, **kwargs) -> Tuple[str, str]:
        """
        Returns the value and standard deviation according to a given measure, calculated as the average across all
        available folds.

        :param measure: The measure
        :return:        A textual representation of the averaged value and the standard deviation
        """
        values = [results[measure] for results in self.results if results]
        values = np.array(values)
        return measure.format(np.average(values), **kwargs), measure.format(np.std(values), **kwargs)

    def avg_dict(self, **kwargs) -> Dict[Measure, str]:
        """
        Returns a dictionary that stores the values, averaged across all folds, as well as the standard deviation,
        according to each measure.

        :return: A dictionary that stores textual representations of the values and standard deviation according to
                 each measure
        """
        result: Dict[Measure, str] = {}

        for measure in self.measures:
            score, std_dev = self.avg(measure, **kwargs)
            result[measure] = score
            result[Measure(measure.option_key, 'Std.-dev. ' + measure.name, measure.percentage)] = std_dev

        return result
