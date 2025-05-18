"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for extracting unique label vectors that are stored as part of rule models.
"""
import logging as log

from typing import List, Optional

import numpy as np

from mlrl.common.cython.output_space_info import LabelVectorSet, LabelVectorSetVisitor, NoOutputSpaceInfo
from mlrl.common.learners import ClassificationRuleLearner

from mlrl.testbed.experiments.dataset_tabular import TabularDataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.label_vectors.label_vector_histogram import LabelVector, LabelVectorHistogram
from mlrl.testbed.experiments.output.label_vectors.label_vectors import LabelVectors
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor
from mlrl.testbed.experiments.state import ExperimentState


class LabelVectorSetExtractor(DataExtractor):
    """
    Allows to extract unique label vectors from a `LabelVectorSet`.
    """

    class Visitor(LabelVectorSetVisitor):
        """
        Accesses the label vectors and frequencies stored by a `LabelVectorSet` and stores them in a
        `LabelVectorHistogram`.
        """

        def __init__(self, num_labels: int):
            """
            :param num_labels: The total number of available labels
            """
            self.label_vector_histogram = LabelVectorHistogram(num_labels=num_labels)

        def visit_label_vector(self, label_vector: np.ndarray, frequency: int):
            """
            See :func:`mlrl.common.cython.output_space_info.LabelVectorSetVisitor.visit_label_vector`
            """
            label_vector = LabelVector(label_indices=label_vector, frequency=frequency)
            self.label_vector_histogram.unique_label_vectors.append(label_vector)

    def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
        """
        See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
        """
        dataset = state.dataset_as(self, TabularDataset)
        learner = state.learner_as(self, ClassificationRuleLearner)

        if dataset and learner:
            output_space_info = learner.output_space_info_

            if isinstance(output_space_info, LabelVectorSet):
                visitor = LabelVectorSetExtractor.Visitor(num_labels=dataset.num_outputs)
                output_space_info.visit(visitor)
                return LabelVectors(visitor.label_vector_histogram)

            if not isinstance(output_space_info, NoOutputSpaceInfo):
                log.error('%s expected type of output space info to be %s, but output space info has type %s',
                          type(self).__name__, LabelVectorSet.__name__,
                          type(output_space_info).__name__)

        return None
