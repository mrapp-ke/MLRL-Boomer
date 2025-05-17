"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for extracting textual representations of rules from rule models.
"""
import logging as log

from typing import List, Optional

from mlrl.common.cython.rule_model import RuleModel
from mlrl.common.mixins import ClassifierMixin, RegressorMixin

from mlrl.testbed.experiments.dataset import TabularDataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.model_text.model_text_rules import RuleModelAsText
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor
from mlrl.testbed.experiments.state import ExperimentState


class RuleModelAsTextExtractor(DataExtractor):
    """
    Allows to extract textual representation of rules from a `RuleModel`.
    """

    @staticmethod
    def __get_model(state: ExperimentState) -> Optional[RuleModel]:
        training_result = state.training_result

        if training_result:
            learner = training_result.learner

            if isinstance(learner, (ClassifierMixin, RegressorMixin)):
                model = learner.model_

                if isinstance(model, RuleModel):
                    return model

                log.error('Cannot handle model of type %s', type(model).__name__)

        return None

    @staticmethod
    def __get_dataset(state: ExperimentState) -> Optional[TabularDataset]:
        dataset = state.dataset

        if isinstance(dataset, TabularDataset):
            return dataset

        log.error('Cannot handle dataset of type %s', type(dataset).__name__)
        return None

    def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
        """
        See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
        """
        model = self.__get_model(state)

        if model:
            dataset = self.__get_dataset(state)

            if dataset:
                return RuleModelAsText(model, dataset)

        return None
