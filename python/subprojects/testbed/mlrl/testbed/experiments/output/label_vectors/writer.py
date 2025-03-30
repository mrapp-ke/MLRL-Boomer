"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing unique label vectors that are contained in a dataset to one or several sinks.
"""
from typing import Optional

import numpy as np

from mlrl.common.cython.output_space_info import LabelVectorSet, LabelVectorSetVisitor
from mlrl.common.learners import ClassificationRuleLearner

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.label_vectors.label_vector_histogram import LabelVectorHistogram
from mlrl.testbed.experiments.output.label_vectors.label_vectors import LabelVectors
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class LabelVectorWriter(OutputWriter):
    """
    Allows to write unique label vectors that are contained in a dataset to one or several sinks.
    """

    class Visitor(LabelVectorSetVisitor):
        """
        Allows to access the label vectors and frequencies store by a `LabelVectorSet`.
        """

        def __init__(self, num_labels: int):
            """
            :param num_labels: The total number of available labels
            """
            self.label_vector_histogram = LabelVectorHistogram(num_labels)

        def visit_label_vector(self, label_vector: np.ndarray, frequency: int):
            """
            See :func:`mlrl.common.cython.output_space_info.LabelVectorSetVisitor.visit_label_vector`
            """
            self.label_vector_histogram.unique_label_vectors.append((label_vector, frequency))

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        dataset = state.dataset
        training_result = state.training_result

        if training_result:
            learner = training_result.learner

            if isinstance(learner, ClassificationRuleLearner):
                output_space_info = learner.output_space_info_

                if isinstance(output_space_info, LabelVectorSet):
                    visitor = LabelVectorWriter.Visitor(num_labels=dataset.num_outputs)
                    output_space_info.visit(visitor)
                    return LabelVectors(visitor.label_vector_histogram)

        return LabelVectors(LabelVectorHistogram.from_dataset(dataset))
