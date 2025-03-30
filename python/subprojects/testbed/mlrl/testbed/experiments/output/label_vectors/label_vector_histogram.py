"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to unique label vectors contained in a dataset.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from scipy.sparse import lil_array

from mlrl.common.config.options import Options
from mlrl.common.data.types import Uint8

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import format_table


class LabelVectorHistogram(TabularOutputData):
    """
    Stores unique label vectors and their respective frequency in a data set.
    """

    OPTION_SPARSE = 'sparse'

    COLUMN_INDEX = 'Index'

    COLUMN_LABEL_VECTOR = 'Label vector'

    COLUMN_FREQUENCY = 'Frequency'

    def __init__(self, num_labels: int, unique_label_vectors: Optional[List[Tuple[np.array, int]]] = None):
        """
        :param num_labels:              The total number of available labels
        :param unique_label_vectors:    A list that contains the unique label vectors, as well as their frequency, or
                                        None if not label vectors should be stored
        """
        super().__init__('Label vectors', 'label_vectors', ExperimentState.FormatterOptions(include_dataset_type=False))
        self.num_labels = num_labels
        self.unique_label_vectors = unique_label_vectors if unique_label_vectors else []

    @staticmethod
    def from_dataset(dataset: Dataset) -> 'LabelVectorHistogram':
        """
        Creates and returns a `LabelVectorHistogram` that stores all unique label vectors contained in a dataset.

        :param dataset: The dataset
        :return:        The `LabelVectorHistogram` that has been created
        """
        unique_label_vector_strings = {}
        separator = ','

        for label_vector in lil_array(dataset.y).rows:
            label_vector_string = separator.join(map(str, label_vector))
            frequency = unique_label_vector_strings.setdefault(label_vector_string, 0)
            unique_label_vector_strings[label_vector_string] = frequency + 1

        unique_label_vectors = []

        for label_vector_string, frequency in unique_label_vector_strings.items():
            label_vector = np.asarray([int(label_index) for label_index in label_vector_string.split(separator)])
            unique_label_vectors.append((label_vector, frequency))

        return LabelVectorHistogram(dataset.num_outputs, unique_label_vectors)

    def __format_label_vector(self, sparse_label_vector: np.ndarray, sparse: bool) -> str:
        if sparse:
            return str(sparse_label_vector)

        dense_label_vector = np.zeros(shape=self.num_labels, dtype=Uint8)
        dense_label_vector[sparse_label_vector] = 1
        return str(dense_label_vector)

    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        sparse = options.get_bool(self.OPTION_SPARSE, False)
        header = [self.COLUMN_INDEX, self.COLUMN_LABEL_VECTOR, self.COLUMN_FREQUENCY]
        rows = []

        for i, (sparse_label_vector, frequency) in enumerate(self.unique_label_vectors):
            rows.append([i + 1, self.__format_label_vector(sparse_label_vector, sparse=sparse), frequency])

        return format_table(rows, header=header)

    def to_table(self, options: Options, **_) -> Optional[TabularOutputData.Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        sparse = options.get_bool(self.OPTION_SPARSE, False)
        rows = []

        for i, (sparse_label_vector, frequency) in enumerate(self.unique_label_vectors):
            rows.append({
                self.COLUMN_INDEX: i + 1,
                self.COLUMN_LABEL_VECTOR: self.__format_label_vector(sparse_label_vector, sparse=sparse),
                self.COLUMN_FREQUENCY: frequency
            })

        return rows
