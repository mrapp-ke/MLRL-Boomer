"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing unique label vectors that are contained in a data set. The label vectors can be written to
one or several outputs, e.g., to the console or to a file.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scipy.sparse import lil_array

from mlrl.common.config.options import Options
from mlrl.common.cython.output_space_info import LabelVectorSet, LabelVectorSetVisitor
from mlrl.common.data.types import Uint8
from mlrl.common.learners import ClassificationRuleLearner

from mlrl.testbed.data_sinks import CsvFileSink as BaseCsvFileSink, LogSink as BaseLogSink
from mlrl.testbed.format import format_table
from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.output_writer import Formattable, OutputWriter, Tabularizable
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.training_result import TrainingResult

OPTION_SPARSE = 'sparse'


class LabelVectorWriter(OutputWriter):
    """
    Allows to write unique label vectors that are contained in a data set to one or several sinks.
    """

    class LabelVectors(Formattable, Tabularizable):
        """
        Stores unique label vectors that are contained in a data set.
        """

        COLUMN_INDEX = 'Index'

        COLUMN_LABEL_VECTOR = 'Label vector'

        COLUMN_FREQUENCY = 'Frequency'

        def __init__(self, num_labels: int, y=None):
            """
            :param num_labels:  The total number of available labels
            :param y:           A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_labels)`, that stores the ground truth labels
            """
            self.num_labels = num_labels

            if y is None:
                self.unique_label_vectors = []
            else:
                unique_label_vector_strings: Dict[str, int] = {}
                y = lil_array(y)
                separator = ','

                for label_vector in y.rows:
                    label_vector_string = separator.join(map(str, label_vector))
                    frequency = unique_label_vector_strings.setdefault(label_vector_string, 0)
                    unique_label_vector_strings[label_vector_string] = frequency + 1

                unique_label_vectors: List[Tuple[np.array, int]] = []

                for label_vector_string, frequency in unique_label_vector_strings.items():
                    label_vector = np.asarray(
                        [int(label_index) for label_index in label_vector_string.split(separator)])
                    unique_label_vectors.append((label_vector, frequency))

                self.unique_label_vectors = unique_label_vectors

        def __format_label_vector(self, sparse_label_vector: np.ndarray, sparse: bool) -> str:
            if sparse:
                return str(sparse_label_vector)

            dense_label_vector = np.zeros(shape=self.num_labels, dtype=Uint8)
            dense_label_vector[sparse_label_vector] = 1
            return str(dense_label_vector)

        def format(self, options: Options, **_) -> str:
            """
            See :func:`mlrl.testbed.output_writer.Formattable.format`
            """
            sparse = options.get_bool(OPTION_SPARSE, False)
            header = [self.COLUMN_INDEX, self.COLUMN_LABEL_VECTOR, self.COLUMN_FREQUENCY]
            rows = []

            for i, (sparse_label_vector, frequency) in enumerate(self.unique_label_vectors):
                rows.append([i + 1, self.__format_label_vector(sparse_label_vector, sparse=sparse), frequency])

            return format_table(rows, header=header)

        def tabularize(self, options: Options, **_) -> Optional[List[Dict[str, str]]]:
            """
            See :func:`mlrl.testbed.output_writer.Tabularizable.tabularize`
            """
            sparse = options.get_bool(OPTION_SPARSE, False)
            rows = []

            for i, (sparse_label_vector, frequency) in enumerate(self.unique_label_vectors):
                columns = {
                    self.COLUMN_INDEX: i + 1,
                    self.COLUMN_LABEL_VECTOR: self.__format_label_vector(sparse_label_vector, sparse=sparse),
                    self.COLUMN_FREQUENCY: frequency
                }
                rows.append(columns)

            return rows

    class LogSink(BaseLogSink):
        """
        Allows to write unique label vectors that are contained in a data set to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(BaseLogSink.TitleFormatter('Label vectors', include_dataset_type=False), options=options)

    class CsvFileSink(BaseCsvFileSink):
        """
        Allows to write unique label vectors that are contained in a data set to a CSV file.
        """

        def __init__(self, directory: str, options: Options = Options()):
            """
            :param directory: The path to the directory, where the CSV file should be located
            """
            super().__init__(BaseCsvFileSink.PathFormatter(directory, 'label_vectors', include_dataset_type=False),
                             options=options)

    # pylint: disable=unused-argument
    def _generate_output_data(self, scope: OutputScope, training_result: Optional[TrainingResult],
                              prediction_result: Optional[PredictionResult]) -> Optional[Any]:
        dataset = scope.dataset
        return LabelVectorWriter.LabelVectors(num_labels=dataset.num_outputs, y=dataset.y)


class LabelVectorSetWriter(LabelVectorWriter):
    """
    Allows to write unique label vectors that are stored as part of a model learned by a rule learning algorithm to one
    or several sinks.
    """

    class Visitor(LabelVectorSetVisitor):
        """
        Allows to access the label vectors and frequencies store by a `LabelVectorSet`.
        """

        def __init__(self, num_labels: int):
            """
            :param num_labels: The total number of available labels
            """
            self.label_vectors = LabelVectorWriter.LabelVectors(num_labels=num_labels)

        def visit_label_vector(self, label_vector: np.ndarray, frequency: int):
            """
            See :func:`mlrl.common.cython.output_space_info.LabelVectorSetVisitor.visit_label_vector`
            """
            self.label_vectors.unique_label_vectors.append((label_vector, frequency))

    def _generate_output_data(self, scope: OutputScope, training_result: Optional[TrainingResult],
                              prediction_result: Optional[PredictionResult]) -> Optional[Any]:
        if training_result:
            learner = training_result.learner

            if isinstance(learner, ClassificationRuleLearner):
                output_space_info = learner.output_space_info_

                if isinstance(output_space_info, LabelVectorSet):
                    visitor = LabelVectorSetWriter.Visitor(num_labels=scope.dataset.num_outputs)
                    output_space_info.visit(visitor)
                    return visitor.label_vectors

        return super()._generate_output_data(scope, training_result, prediction_result)
