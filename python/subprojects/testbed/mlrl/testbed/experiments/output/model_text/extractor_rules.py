"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for extracting textual representations of rules from rule models.
"""
import logging as log

from typing import List, Optional

from mlrl.common.cython.rule_model import RuleModel
from mlrl.common.mixins import ClassifierMixin, RegressorMixin

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
        training_result = state.training_result
        dataset = state.dataset

        if training_result and dataset:
            learner = training_result.learner

            if isinstance(learner, (ClassifierMixin, RegressorMixin)):
                model = learner.model_

                if isinstance(model, RuleModel):
                    return RuleModelAsText(model, dataset)

                log.error('Cannot handle model of type %s', type(model).__name__)

        return None
