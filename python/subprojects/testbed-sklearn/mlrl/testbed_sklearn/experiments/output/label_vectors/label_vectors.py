"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing unique label vectors contained in a dataset that are part of output data.
"""
from typing import List, Optional, Tuple, override

from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vector_histogram import LabelVectorHistogram

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import TabularProperties
from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.table import RowWiseTable, Table

from mlrl.util.options import Options


class LabelVectors(TabularOutputData):
    """
    Represents unique label vectors contained in a dataset that are part of output data.
    """

    PROPERTIES = TabularProperties(name='Label vectors', file_name='label_vectors')

    CONTEXT = Context(include_dataset_type=False)

    COLUMN_INDEX = 'Index'

    COLUMN_LABEL_VECTOR = 'Label vector'

    COLUMN_FREQUENCY = 'Frequency'

    def __init__(self, values: List[Tuple[str, int]]):
        """
        :param values: A list that stores textual representations of unique label vectors and their respective frequency
        """
        super().__init__(properties=self.PROPERTIES, context=self.CONTEXT)
        self.values = values

    @staticmethod
    def from_histogram(histogram: LabelVectorHistogram) -> 'LabelVectors':
        """
        Creates and returns `LabelVectors` from a given histogram.

        :param histogram:   The histogram
        :return:            The `LabelVectors` that have been created
        """
        values = [(str(label_vector), label_vector.frequency) for label_vector in histogram.unique_label_vectors]
        return LabelVectors(values)

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        table = self.to_table(options, **kwargs)
        return table.format() if table else None

    # pylint: disable=unused-argument
    @override
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        table = RowWiseTable(self.COLUMN_INDEX, self.COLUMN_LABEL_VECTOR, self.COLUMN_FREQUENCY)

        for i, (label_vector, frequency) in enumerate(self.values):
            table.add_row(i + 1, label_vector, frequency)

        return table.sort_by_columns(2, descending=True)
