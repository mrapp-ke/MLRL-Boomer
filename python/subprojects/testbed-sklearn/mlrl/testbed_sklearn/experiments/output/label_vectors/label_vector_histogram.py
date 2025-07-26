"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to unique label vectors.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, override

import numpy as np

from scipy.sparse import lil_array

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset


@dataclass
class LabelVector:
    """
    A label vector.

    Attributes:
        label_indices:  The indices of all relevant labels in the label vector
        frequency:      The frequency of the label vector
    """
    label_indices: np.ndarray
    frequency: int

    @override
    def __str__(self) -> str:
        return str(self.label_indices)


class LabelVectorHistogram:
    """
    Stores unique label vectors and their respective frequency.
    """

    def __init__(self, num_labels: int, unique_label_vectors: Optional[List[LabelVector]] = None):
        """
        :param num_labels:              The total number of available labels
        :param unique_label_vectors:    A list that contains the unique label vectors, as well as their frequency, or
                                        None if not label vectors should be stored
        """
        self.num_labels = num_labels
        self.unique_label_vectors = unique_label_vectors if unique_label_vectors else []

    @staticmethod
    def from_dataset(dataset: TabularDataset) -> 'LabelVectorHistogram':
        """
        Creates and returns a `LabelVectorHistogram` that stores all unique label vectors contained in a dataset.

        :param dataset: The dataset
        :return:        The `LabelVectorHistogram` that has been created
        """
        unique_label_vector_strings: Dict[str, int] = {}
        separator = ','

        for label_vector in lil_array(dataset.y).rows:
            label_vector_string = separator.join(map(str, label_vector))
            frequency = unique_label_vector_strings.setdefault(label_vector_string, 0)
            unique_label_vector_strings[label_vector_string] = frequency + 1

        unique_label_vectors = []

        for label_vector_string, frequency in unique_label_vector_strings.items():
            label_vector = np.asarray([int(label_index) for label_index in label_vector_string.split(separator)])
            unique_label_vectors.append(LabelVector(label_indices=label_vector, frequency=frequency))

        return LabelVectorHistogram(dataset.num_outputs, unique_label_vectors)
