"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for extracting textual representations of rules from rule models.
"""
import logging as log

from typing import List, Optional

from mlrl.common.cython.rule_model import RuleModel
from mlrl.common.mixins import ClassifierMixin, RegressorMixin

from mlrl.testbed.experiments.dataset_tabular import TabularDataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.model_text.model_text_rules import RuleModelAsText
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor
from mlrl.testbed.experiments.state import ExperimentState


class RuleModelAsTextExtractor(DataExtractor):
    """
    Allows to extract textual representation of rules from a `RuleModel`.
    """

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
