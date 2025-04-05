"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of models. The characteristics can be written to one or several
outputs, e.g., to the console or to a file.
"""
import logging as log

from typing import Optional

import numpy as np

from mlrl.common.cython.rule_model import CompleteHead, ConjunctiveBody, EmptyBody, PartialHead, RuleModel, \
    RuleModelVisitor
from mlrl.common.mixins import ClassifierMixin, RegressorMixin

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.models.characteristics_rules import RuleModelCharacteristics
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class RuleModelCharacteristicsWriter(OutputWriter):
    """
    Allows to write the characteristics of a `RuleModel` to one or several sinks.
    """

    class RuleModelCharacteristicsVisitor(RuleModelVisitor):
        """
        A visitor that allows to determine the characteristics of a `RuleModel`.
        """

        def __init__(self):
            self.num_numerical_leq = []
            self.num_numerical_gr = []
            self.num_ordinal_leq = []
            self.num_ordinal_gr = []
            self.num_nominal_eq = []
            self.num_nominal_neq = []
            self.num_pos_predictions = []
            self.num_neg_predictions = []
            self.default_rule_index = None
            self.default_rule_pos_predictions = 0
            self.default_rule_neg_predictions = 0
            self.index = -1

        def visit_empty_body(self, _: EmptyBody):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_empty_body`
            """
            self.index += 1
            self.default_rule_index = self.index

        def visit_conjunctive_body(self, body: ConjunctiveBody):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_conjunctive_body`
            """
            self.index += 1
            self.num_numerical_leq.append(
                body.numerical_leq_indices.shape[0] if body.numerical_leq_indices is not None else 0)
            self.num_numerical_gr.append(
                body.numerical_gr_indices.shape[0] if body.numerical_gr_indices is not None else 0)
            self.num_ordinal_leq.append(
                body.ordinal_leq_indices.shape[0] if body.ordinal_leq_indices is not None else 0)
            self.num_ordinal_gr.append(body.ordinal_gr_indices.shape[0] if body.ordinal_gr_indices is not None else 0)
            self.num_nominal_eq.append(body.nominal_eq_indices.shape[0] if body.nominal_eq_indices is not None else 0)
            self.num_nominal_neq.append(
                body.nominal_neq_indices.shape[0] if body.nominal_neq_indices is not None else 0)

        def visit_complete_head(self, head: CompleteHead):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_complete_head`
            """
            num_pos_predictions = np.count_nonzero(head.scores > 0)
            num_neg_predictions = head.scores.shape[0] - num_pos_predictions

            if self.index == self.default_rule_index:
                self.default_rule_pos_predictions = num_pos_predictions
                self.default_rule_neg_predictions = num_neg_predictions
            else:
                self.num_pos_predictions.append(num_pos_predictions)
                self.num_neg_predictions.append(num_neg_predictions)

        def visit_partial_head(self, head: PartialHead):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_partial_head`
            """
            num_pos_predictions = np.count_nonzero(head.scores > 0)
            num_neg_predictions = head.scores.shape[0] - num_pos_predictions

            if self.index == self.default_rule_index:
                self.default_rule_pos_predictions = num_pos_predictions
                self.default_rule_neg_predictions = num_neg_predictions
            else:
                self.num_pos_predictions.append(num_pos_predictions)
                self.num_neg_predictions.append(num_neg_predictions)

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        training_result = state.training_result

        if training_result:
            learner = training_result.learner

            if isinstance(learner, (ClassifierMixin, RegressorMixin)):
                model = learner.model_

                if isinstance(model, RuleModel):
                    visitor = RuleModelCharacteristicsWriter.RuleModelCharacteristicsVisitor()
                    model.visit_used(visitor)
                    return RuleModelCharacteristics(default_rule_index=visitor.default_rule_index,
                                                    default_rule_pos_predictions=visitor.default_rule_pos_predictions,
                                                    default_rule_neg_predictions=visitor.default_rule_neg_predictions,
                                                    num_numerical_leq=np.asarray(visitor.num_numerical_leq),
                                                    num_numerical_gr=np.asarray(visitor.num_numerical_gr),
                                                    num_ordinal_leq=np.asarray(visitor.num_ordinal_leq),
                                                    num_ordinal_gr=np.asarray(visitor.num_ordinal_gr),
                                                    num_nominal_eq=np.asarray(visitor.num_nominal_eq),
                                                    num_nominal_neq=np.asarray(visitor.num_nominal_neq),
                                                    num_pos_predictions=np.asarray(visitor.num_pos_predictions),
                                                    num_neg_predictions=np.asarray(visitor.num_neg_predictions))

            log.error('The learner does not support to obtain model characteristics')

        return None
