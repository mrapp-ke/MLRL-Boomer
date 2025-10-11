"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write label vectors to one or several sinks.
"""
import logging as log

from argparse import Namespace
from typing import List, Optional, Set, override

import numpy as np

from mlrl.common.cython.output_space_info import LabelVectorSet, LabelVectorSetVisitor, NoOutputSpaceInfo
from mlrl.common.learners import ClassificationRuleLearner

from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vector_histogram import LabelVector, \
    LabelVectorHistogram
from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vectors import LabelVectors

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor
from mlrl.testbed.experiments.state import ExperimentMode, ExperimentState
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument


class LabelVectorSetExtension(Extension):
    """
    An extension that configures the functionality to write label vectors, extracted from a `LabelVectorSet`, to one or
    several sinks.
    """

    class LabelVectorSetExtractor(DataExtractor):
        """
        Allows to extract unique label vectors from a `LabelVectorSet`.
        """

        class Visitor(LabelVectorSetVisitor):
            """
            Accesses the label vectors and frequencies stored by a `LabelVectorSet` and stores them in a
            `LabelVectorHistogram`.
            """

            def __init__(self):
                self.label_vector_histogram = LabelVectorHistogram()

            @override
            def visit_label_vector(self, label_vector: np.ndarray, frequency: int):
                """
                See :func:`mlrl.common.cython.output_space_info.LabelVectorSetVisitor.visit_label_vector`
                """
                label_vector = LabelVector(label_indices=label_vector, frequency=frequency)
                self.label_vector_histogram.unique_label_vectors.append(label_vector)

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            learner = state.learner_as(self, ClassificationRuleLearner)

            if learner:
                output_space_info = learner.output_space_info_

                if isinstance(output_space_info, LabelVectorSet):
                    visitor = LabelVectorSetExtension.LabelVectorSetExtractor.Visitor()
                    output_space_info.visit(visitor)
                    return LabelVectors.from_histogram(visitor.label_vector_histogram)

                if not isinstance(output_space_info, NoOutputSpaceInfo):
                    log.error('%s expected type of output space info to be %s, but output space info has type %s',
                              type(self).__name__, LabelVectorSet.__name__,
                              type(output_space_info).__name__)

            return None

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), ResultDirectoryExtension(), *dependencies)

    @override
    def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return set()

    # pylint: disable=unused-argument
    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, mode: ExperimentMode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        experiment_builder.label_vector_writer.extractors.insert(1, LabelVectorSetExtension.LabelVectorSetExtractor())

    @override
    def get_supported_modes(self) -> Set[ExperimentMode]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {ExperimentMode.SINGLE, ExperimentMode.BATCH, ExperimentMode.READ, ExperimentMode.RUN}
