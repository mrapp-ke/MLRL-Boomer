"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for extracting characteristics from rule models.
"""
import logging as log

from typing import List, Optional

import numpy as np

from mlrl.common.cython.rule_model import CompleteHead, ConjunctiveBody, EmptyBody, PartialHead, RuleModel, \
    RuleModelVisitor
from mlrl.common.mixins import ClassifierMixin, RegressorMixin

from mlrl.testbed.experiments.output.characteristics.model.characteristics_rules import BodyStatistics, \
    HeadStatistics, RuleModelCharacteristics, RuleModelStatistics, RuleStatistics
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor
from mlrl.testbed.experiments.state import ExperimentState


class RuleModelCharacteristicsExtractor(DataExtractor):
    """
    Allows to extract characteristics from a `RuleModel`.
    """

    class Visitor(RuleModelVisitor):
        """
        A visitor that allows to determine the characteristics of a `RuleModel`.
        """

        def __init__(self):
            self.default_rule_index = None
            self.index = -1
            self.statistics = RuleModelStatistics()

        def visit_empty_body(self, _: EmptyBody):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_empty_body`
            """
            self.index += 1
            self.default_rule_index = self.index
            self.statistics.default_rule_statistics = RuleStatistics(body_statistics=BodyStatistics())

        def visit_conjunctive_body(self, body: ConjunctiveBody):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_conjunctive_body`
            """
            self.index += 1
            body_statistics = BodyStatistics(
                num_numerical_leq=body.numerical_leq_indices.size if body.numerical_leq_indices is not None else 0,
                num_numerical_gr=body.numerical_gr_indices.size if body.numerical_gr_indices is not None else 0,
                num_ordinal_leq=body.ordinal_leq_indices.size if body.ordinal_leq_indices is not None else 0,
                num_ordinal_gr=body.ordinal_gr_indices.size if body.ordinal_gr_indices is not None else 0,
                num_nominal_eq=body.nominal_eq_indices.size if body.nominal_eq_indices is not None else 0,
                num_nominal_neq=body.nominal_neq_indices.size if body.nominal_neq_indices is not None else 0)
            self.statistics.rule_statistics.append(RuleStatistics(body_statistics=body_statistics))

        def visit_complete_head(self, head: CompleteHead):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_complete_head`
            """
            num_positive_predictions = np.count_nonzero(head.scores > 0)
            num_negative_predictions = head.scores.shape[0] - num_positive_predictions
            head_statistics = HeadStatistics(num_positive_predictions=num_positive_predictions,
                                             num_negative_predictions=num_negative_predictions)

            if self.index == self.default_rule_index:
                self.statistics.default_rule_statistics.head_statistics = head_statistics
            else:
                self.statistics.rule_statistics[-1].head_statistics = head_statistics

        def visit_partial_head(self, head: PartialHead):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_partial_head`
            """
            num_positive_predictions = np.count_nonzero(head.scores > 0)
            num_negative_predictions = head.scores.shape[0] - num_positive_predictions
            head_statistics = HeadStatistics(num_positive_predictions=num_positive_predictions,
                                             num_negative_predictions=num_negative_predictions)

            if self.index == self.default_rule_index:
                self.statistics.default_rule_statistics.head_statistics = head_statistics
            else:
                self.statistics.rule_statistics[-1].head_statistics = head_statistics

    def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
        """
        See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
        """
        training_result = state.training_result

        if training_result:
            learner = training_result.learner

            if isinstance(learner, (ClassifierMixin, RegressorMixin)):
                model = learner.model_

                if isinstance(model, RuleModel):
                    visitor = RuleModelCharacteristicsExtractor.Visitor()
                    model.visit_used(visitor)
                    return RuleModelCharacteristics(visitor.statistics)

                log.error('Cannot handle model of type %s', type(model).__name__)

        return None
