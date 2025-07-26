"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing unique label vectors contained in a dataset that are part of output data.
"""
from typing import Optional, override

import numpy as np

from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vector_histogram import LabelVector, \
    LabelVectorHistogram

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.output.data import OutputData, TabularOutputData
from mlrl.testbed.experiments.table import RowWiseTable, Table

from mlrl.util.options import Options


class LabelVectors(TabularOutputData):
    """
    Represents unique label vectors contained in a dataset that are part of output data.
    """

    OPTION_SPARSE = 'sparse'

    def __init__(self, label_vector_histogram: LabelVectorHistogram):
        """
        :param label_vector_histogram: The histogram that stores unique label vectors and their respective frequency
        """
        super().__init__(OutputData.Properties(name='Label vectors', file_name='label_vectors'),
                         Context(include_dataset_type=False))
        self.label_vector_histogram = label_vector_histogram

    def __format_label_vector(self, label_vector: LabelVector, sparse: bool) -> str:
        if sparse:
            return str(label_vector)

        dense_label_vector = np.zeros(shape=self.label_vector_histogram.num_labels, dtype=np.uint8)
        dense_label_vector[label_vector.label_indices] = 1
        return str(dense_label_vector)

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        table = self.to_table(options, **kwargs)
        return table.format() if table else None

    @override
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        sparse = options.get_bool(self.OPTION_SPARSE, kwargs.get(self.OPTION_SPARSE, False))
        table = RowWiseTable('Index', 'Label vector', 'Frequency')

        for i, label_vector in enumerate(self.label_vector_histogram.unique_label_vectors):
            table.add_row(i + 1, self.__format_label_vector(label_vector, sparse=sparse), label_vector.frequency)

        return table.sort_by_columns(2, descending=True)
