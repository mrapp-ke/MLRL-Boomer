"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions to determine certain characteristics of feature or output matrices.
"""
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np

from mlrl.common.arrays import is_sparse
from mlrl.common.options import Options

from mlrl.testbed.format import OPTION_DECIMALS, OPTION_PERCENTAGE, Formatter, filter_formatters, format_table
from mlrl.testbed.output_writer import Formattable, Tabularizable
from mlrl.testbed.problem_type import ProblemType

OPTION_OUTPUTS = 'outputs'

OPTION_OUTPUT_DENSITY = 'output_density'

OPTION_OUTPUT_SPARSITY = 'output_sparsity'

OPTION_LABEL_IMBALANCE_RATIO = 'label_imbalance_ratio'

OPTION_LABEL_CARDINALITY = 'label_cardinality'

OPTION_DISTINCT_LABEL_VECTORS = 'distinct_label_vectors'


def density(matrix) -> float:
    """
    Calculates and returns the density of a given feature or output matrix.

    :param matrix:  A `numpy.ndarray` or `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_rows, num_cols)`, that stores the feature values of training examples or their ground truth
    :return:        The fraction of dense elements explicitly stored in the given matrix among all elements
    """
    num_elements = matrix.shape[0] * matrix.shape[1]

    if is_sparse(matrix):
        num_dense_elements = matrix.nnz
    else:
        num_dense_elements = np.count_nonzero(matrix)

    return num_dense_elements / num_elements if num_elements > 0 else 0


def label_cardinality(y) -> float:
    """
    Calculates and returns the average label cardinality of a given label matrix.

    :param y:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                `(num_examples, num_labels)`, that stores the labels of training examples
    :return:    The average number of relevant labels per training example
    """
    if is_sparse(y):
        y = y.tocsr()
        num_relevant_per_example = y.indptr[1:] - y.indptr[:-1]
    else:
        num_relevant_per_example = np.count_nonzero(y, axis=1)

    return np.average(num_relevant_per_example)


def distinct_label_vectors(y) -> int:
    """
    Determines and returns the number of distinct label vectors in a label matrix.

    :param y:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                `(num_examples, num_labels)`, that stores the labels of training examples
    :return:    The number of distinct label vectors in the given matrix
    """
    if is_sparse(y):
        y = y.tolil()
        return np.unique(y.rows).shape[0]

    return np.unique(y, axis=0).shape[0]


def label_imbalance_ratio(y) -> float:
    """
    Calculates and returns the average label imbalance ratio of a given label matrix.

    :param y:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                `(num_examples, num_labels)`, that stores the labels of training examples
    :return:    The label imbalance ratio averaged over the available labels
    """
    if is_sparse(y):
        y = y.tocsc()
        num_relevant_per_label = y.indptr[1:] - y.indptr[:-1]
    else:
        num_relevant_per_label = np.count_nonzero(y, axis=0)

    num_relevant_per_label = num_relevant_per_label[num_relevant_per_label != 0]

    if num_relevant_per_label.shape[0] > 0:
        return np.average(np.max(num_relevant_per_label) / num_relevant_per_label)

    return 0.0


class OutputCharacteristics(Formattable, Tabularizable):
    """
    Stores characteristics of an output matrix.
    """

    def __init__(self, y, problem_type: ProblemType):
        """
        :param y:               A `numpy.ndarray`, `scipy.sparse.spmatrix`, `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the ground truth
        :param problem_type:    The type of the machine learning problem
        """
        self._y = y
        self.num_outputs = y.shape[1]
        classification = problem_type == ProblemType.CLASSIFICATION
        self.formatters = LABEL_CHARACTERISTICS if classification else OUTPUT_CHARACTERISTICS

    @cached_property
    def output_density(self):
        """
        The density of the output matrix.
        """
        return density(self._y)

    @property
    def output_sparsity(self):
        """
        The sparsity of the output matrix.
        """
        return 1 - self.output_density

    @cached_property
    def avg_label_imbalance_ratio(self):
        """
        The average label imbalance ratio of the label matrix.
        """
        return label_imbalance_ratio(self._y)

    @cached_property
    def avg_label_cardinality(self):
        """
        The average label cardinality of the label matrix.
        """
        return label_cardinality(self._y)

    @cached_property
    def num_distinct_label_vectors(self):
        """
        The number of distinct label vectors in the label matrix.
        """
        return distinct_label_vectors(self._y)

    def format(self, options: Options, **_) -> str:
        """
        See :func:`mlrl.testbed.output_writer.Formattable.format`
        """
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 2)
        rows = []

        for formatter in filter_formatters(self.formatters, [options]):
            rows.append([formatter.name, formatter.format(self, percentage=percentage, decimals=decimals)])

        return format_table(rows)

    def tabularize(self, options: Options, **_) -> Optional[List[Dict[str, str]]]:
        """
        See :func:`mlrl.testbed.output_writer.Tabularizable.tabularize`
        """
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 0)
        columns = {}

        for formatter in filter_formatters(self.formatters, [options]):
            columns[formatter] = formatter.format(self, percentage=percentage, decimals=decimals)

        return [columns]


class Characteristic(Formatter):
    """
    Allows to create textual representations of characteristics.
    """

    def __init__(self, option: str, name: str, getter_function, percentage: bool = False):
        """
        :param getter_function: The getter function that should be used to retrieve the characteristic
        """
        super().__init__(option, name, percentage)
        self.getter_function = getter_function

    def format(self, value, **kwargs) -> str:
        """
        See :func:`mlrl.testbed.output_writer.Formattable.format`
        """
        return super().format(self.getter_function(value), **kwargs)


OUTPUT_CHARACTERISTICS: List[Characteristic] = [
    Characteristic(OPTION_OUTPUTS, 'Outputs', lambda x: x.num_outputs),
    Characteristic(OPTION_OUTPUT_DENSITY, 'Output Density', lambda x: x.output_density, percentage=True),
    Characteristic(OPTION_OUTPUT_SPARSITY, 'Output Sparsity', lambda x: x.output_sparsity, percentage=True)
]

LABEL_CHARACTERISTICS: List[Characteristic] = OUTPUT_CHARACTERISTICS + [
    Characteristic(OPTION_LABEL_IMBALANCE_RATIO, 'Label Imbalance Ratio', lambda x: x.avg_label_imbalance_ratio),
    Characteristic(OPTION_LABEL_CARDINALITY, 'Label Cardinality', lambda x: x.avg_label_cardinality),
    Characteristic(OPTION_DISTINCT_LABEL_VECTORS, 'Distinct Label Vectors', lambda x: x.num_distinct_label_vectors)
]
