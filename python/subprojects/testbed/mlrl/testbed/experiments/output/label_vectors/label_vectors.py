"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing unique label vectors contained in a dataset that are part of output data.
"""
from typing import Optional

import numpy as np

from mlrl.common.config.options import Options
from mlrl.common.data.types import Uint8

from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.output.label_vectors.label_vector_histogram import LabelVector, LabelVectorHistogram
from mlrl.testbed.experiments.output.table import RowWiseTable, Table


class LabelVectors(TabularOutputData):
    """
    Represents unique label vectors contained in a dataset that are part of output data.
    """

    OPTION_SPARSE = 'sparse'

    def __init__(self, label_vector_histogram: LabelVectorHistogram):
        """
        :param label_vector_histogram: The histogram that stores unique label vectors and their respective frequency
        """
        super().__init__(name='Label vectors',
                         file_name='label_vectors',
                         default_context=Data.Context(include_dataset_type=False))
        self.label_vector_histogram = label_vector_histogram

    def __format_label_vector(self, label_vector: LabelVector, sparse: bool) -> str:
        if sparse:
            return str(label_vector)

        dense_label_vector = np.zeros(shape=self.label_vector_histogram.num_labels, dtype=Uint8)
        dense_label_vector[label_vector.label_indices] = 1
        return str(dense_label_vector)

    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        return self.to_table(options, **kwargs).format()

    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        sparse = options.get_bool(self.OPTION_SPARSE, kwargs.get(self.OPTION_SPARSE, False))
        table = RowWiseTable('Index', 'Label vector', 'Frequency')

        for i, label_vector in enumerate(self.label_vector_histogram.unique_label_vectors):
            table.add_row(i + 1, self.__format_label_vector(label_vector, sparse=sparse), label_vector.frequency)

        return table.sort_by_columns(2, descending=True)
