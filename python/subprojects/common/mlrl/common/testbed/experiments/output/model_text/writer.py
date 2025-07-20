"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing textual representations of models to one or several sinks.
"""
import logging as log

from typing import List, Optional, override

from mlrl.common.cython.rule_model import RuleModel
from mlrl.common.mixins import ClassifierMixin, RegressorMixin
from mlrl.common.testbed.experiments.output.model_text.model_text import RuleModelAsText

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class RuleModelAsTextWriter(OutputWriter):
    """
    Allows to write textual representations of models to one or several sinks.
    """

    class DefaultExtractor(DataExtractor):
        """
        Allows to extract textual representation of rules from a `RuleModel`.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            dataset = state.dataset_as(self, TabularDataset)
            learner = state.learner_as(self, ClassifierMixin, RegressorMixin)

            if dataset and learner:
                model = learner.model_

                if isinstance(model, RuleModel):
                    return RuleModelAsText(model, dataset)

                log.error('%s expected type of model to be %s, but model has type %s',
                          type(self).__name__, RuleModel.__name__,
                          type(model).__name__)

            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, RuleModelAsTextWriter.DefaultExtractor())
