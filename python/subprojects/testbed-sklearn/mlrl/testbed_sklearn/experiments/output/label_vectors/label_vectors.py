"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing unique label vectors contained in a dataset that are part of output data.
"""
from typing import Optional, override

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

    def __init__(self, label_vector_histogram: LabelVectorHistogram):
        """
        :param label_vector_histogram: The histogram that stores unique label vectors and their respective frequency
        """
        super().__init__(properties=self.PROPERTIES, context=self.CONTEXT)
        self.label_vector_histogram = label_vector_histogram

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
        table = RowWiseTable('Index', 'Label vector', 'Frequency')

        for i, label_vector in enumerate(self.label_vector_histogram.unique_label_vectors):
            table.add_row(i + 1, str(label_vector), label_vector.frequency)

        return table.sort_by_columns(2, descending=True)
