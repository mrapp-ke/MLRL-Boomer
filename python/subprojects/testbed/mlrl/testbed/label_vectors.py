"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing unique label vectors that are contained in a data set. The label vectors can be written to
one or several outputs, e.g., to the console or to a file.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scipy.sparse import lil_matrix

from mlrl.common.options import Options

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import format_table
from mlrl.testbed.output_writer import Formattable, OutputWriter, Tabularizable
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType


class LabelVectorWriter(OutputWriter):
    """
    Allows to write unique label vectors that are contained in a data set to one or severals sinks.
    """

    class LabelVectors(Formattable, Tabularizable):
        """
        Stores unique label vectors that are contained in a data set.
        """

        def __init__(self, y):
            """
            :param y: A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                      ground truth labels
            """
            unique_label_vector_strings: Dict[str, int] = {}
            y = lil_matrix(y)
            separator = ','

            for label_vector in y.rows:
                label_vector_string = separator.join(map(str, label_vector))
                frequency = unique_label_vector_strings.setdefault(label_vector_string, 0)
                unique_label_vector_strings[label_vector_string] = frequency + 1

            unique_label_vectors: List[Tuple[np.array, int]] = []

            for label_vector_string, frequency in unique_label_vector_strings.items():
                label_vector = np.asarray([int(label_index) for label_index in label_vector_string.split(separator)])
                unique_label_vectors.append((label_vector, frequency))

            self.unique_label_vectors = unique_label_vectors

        def format(self, options: Options, **kwargs) -> str:
            header = ['Index', 'Label vector', 'Frequency']
            rows = []

            for i, (label_vector, frequency) in enumerate(self.unique_label_vectors):
                rows.append([i, str(label_vector), frequency])

            return format_table(rows, header=header)

        def tabularize(self, options: Options, **kwargs) -> List[Dict[str, str]]:
            rows = []

            for i, (label_vector, frequency) in enumerate(self.unique_label_vectors):
                columns = {'Index': i, 'Label vector': str(label_vector), 'Frequency': frequency}
                rows.append(columns)

            return rows

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write unique label vectors that are contained in a data set to the console.
        """

        def __init__(self):
            super().__init__(title='Label vectors')

    class CsvSink(OutputWriter.CsvSink):
        """
        Allows to write unique label vectors that are contained in a data set to a CSV file.
        """

        def __init__(self, output_dir: str):
            super().__init__(output_dir=output_dir, file_name='label_vectors')

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)

    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        return LabelVectorWriter.LabelVectors(y)
