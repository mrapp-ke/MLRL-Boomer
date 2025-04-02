"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing unique label vectors contained in a dataset that are part of output data.
"""
from typing import Optional

import numpy as np

from mlrl.common.config.options import Options
from mlrl.common.data.types import Uint8

from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.output.label_vectors.label_vector_histogram import LabelVectorHistogram
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import format_table


class LabelVectors(TabularOutputData):
    """
    Represents unique label vectors contained in a dataset that are part of output data.
    """

    OPTION_SPARSE = 'sparse'

    COLUMN_INDEX = 'Index'

    COLUMN_LABEL_VECTOR = 'Label vector'

    COLUMN_FREQUENCY = 'Frequency'

    def __init__(self, label_vector_histogram: LabelVectorHistogram):
        """
        :param label_vector_histogram: The histogram that stores unique label vectors and their respective frequency
        """
        super().__init__('Label vectors', 'label_vectors', ExperimentState.FormatterOptions(include_dataset_type=False))
        self.label_vector_histogram = label_vector_histogram

    def __format_label_vector(self, label_vector: np.ndarray, sparse: bool) -> str:
        if sparse:
            return str(label_vector)

        dense_label_vector = np.zeros(shape=self.label_vector_histogram.num_labels, dtype=Uint8)
        dense_label_vector[label_vector] = 1
        return str(dense_label_vector)

    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        sparse = options.get_bool(self.OPTION_SPARSE, False)
        header = [self.COLUMN_INDEX, self.COLUMN_LABEL_VECTOR, self.COLUMN_FREQUENCY]
        rows = []

        for i, (label_vector, frequency) in enumerate(self.label_vector_histogram.unique_label_vectors):
            rows.append([i + 1, self.__format_label_vector(label_vector, sparse=sparse), frequency])

        return format_table(rows, header=header)

    def to_table(self, options: Options, **_) -> Optional[TabularOutputData.Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        sparse = options.get_bool(self.OPTION_SPARSE, False)
        rows = []

        for i, (label_vector, frequency) in enumerate(self.label_vector_histogram.unique_label_vectors):
            rows.append({
                self.COLUMN_INDEX: i + 1,
                self.COLUMN_LABEL_VECTOR: self.__format_label_vector(label_vector, sparse=sparse),
                self.COLUMN_FREQUENCY: frequency
            })

        return rows
