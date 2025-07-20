"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for keeping track of several measurements according to different measures.
"""

from typing import Dict, Tuple

import numpy as np

from mlrl.testbed.experiments.output.data import OutputValue
from mlrl.testbed.experiments.output.evaluation.measures import Measure


class Measurements:
    """
    Keeps track of values that correspond to different measures.
    """

    def __init__(self, num_values_per_measure: int):
        """
        :param num_values_per_measure: The number of values to be tracked of for each measure
        """
        self.num_values_per_measure = num_values_per_measure
        self._values_per_measure: Dict[OutputValue, np.ndarray] = {}

    def values_by_measure(self, measure: Measure) -> np.ndarray:
        """
        Returns an array that stores the values that have been tracked for a given measure.

        :param measure: The measure
        :return:        A `np.ndarray`, shape `(num_values_per_measure)`, that stores the values for the given measure
        """
        return self._values_per_measure.setdefault(
            measure, np.full(shape=self.num_values_per_measure, dtype=float, fill_value=np.nan))

    def average_by_measure(self, measure: OutputValue) -> Tuple[float, float]:
        """
        Returns an average and a corresponding standard deviation for a given measure. The average is calculated as the
        arithmetic mean of all values that have been tracked for the measure.

        :param measure: The measure
        :return:        An average and a corresponding standard deviation
        """
        values = self._values_per_measure[measure]
        return np.average(values), np.std(values)

    def values_as_dict(self, index: int) -> Dict[OutputValue, float]:
        """
        Returns a dictionary that contains the value at a specific index that has been tracked for each measure.

        :param index:   The index of the value that should be returned for each measure
        :return:        A dictionary that stores a value for each measure
        """
        return {measure: values[index] for measure, values in self._values_per_measure.items()}

    def averages_as_dict(self) -> Dict[OutputValue, float]:
        """
        Returns a dictionary that stores an average and a corresponding standard deviation for each measure. The
        averages are calculated as the arithmetic mean of all values that have been tracked for an individual measure.

        :return: A dictionary that stores an average and a corresponding standard deviation for each measure
        """
        result = {}

        for measure in self._values_per_measure:
            average, std_dev = self.average_by_measure(measure)
            result[measure] = average
            result[OutputValue(measure.option_key, 'Std.-dev. ' + measure.name, measure.percentage)] = std_dev

        return result
