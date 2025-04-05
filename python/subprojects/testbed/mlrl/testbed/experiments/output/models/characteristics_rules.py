"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of rule models that are part of output data.
"""

from typing import Optional

import numpy as np

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import format_number, format_percentage, format_table


class RuleModelCharacteristics(TabularOutputData):
    """
    Represents characteristics of a rule model that are part of output data.
    """

    def __init__(self, default_rule_index: int, default_rule_pos_predictions: int, default_rule_neg_predictions: int,
                 num_numerical_leq: np.ndarray, num_numerical_gr: np.ndarray, num_ordinal_leq: np.ndarray,
                 num_ordinal_gr: np.ndarray, num_nominal_eq: np.ndarray, num_nominal_neq: np.ndarray,
                 num_pos_predictions: np.ndarray, num_neg_predictions: np.ndarray):
        """
        :param default_rule_index:              The index of the default rule or None, if no default rule is used
        :param default_rule_pos_predictions:    The number of positive predictions of the default rule, if any
        :param default_rule_neg_predictions:    The number of negative predictions of the default rule, if any
        :param num_numerical_leq:               A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                numerical conditions that use the <= operator per rule
        :param num_numerical_gr:                A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                numerical conditions that use the > operator per rule
        :param num_ordinal_leq:                 A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                ordinal conditions that use the <= operator per rule
        :param num_ordinal_gr:                  A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                ordinal conditions that use the > operator per rule
        :param num_nominal_eq:                  A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                nominal conditions that use the == operator per rule
        :param num_nominal_neq:                 A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                nominal conditions that use the != operator per rule
        :param num_pos_predictions:             A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                positive predictions per rule
        :param num_neg_predictions:             A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                negative predictions per rule
        """
        super().__init__('Model characteristics', 'model_characteristics',
                         ExperimentState.FormatterOptions(include_dataset_type=False))
        self.default_rule_index = default_rule_index
        self.default_rule_pos_predictions = default_rule_pos_predictions
        self.default_rule_neg_predictions = default_rule_neg_predictions
        self.num_numerical_leq = num_numerical_leq
        self.num_numerical_gr = num_numerical_gr
        self.num_ordinal_leq = num_ordinal_leq
        self.num_ordinal_gr = num_ordinal_gr
        self.num_nominal_eq = num_nominal_eq
        self.num_nominal_neq = num_nominal_neq
        self.num_pos_predictions = num_pos_predictions
        self.num_neg_predictions = num_neg_predictions

    # pylint: disable=unused-argument
    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        num_predictions = self.num_pos_predictions + self.num_neg_predictions
        num_conditions = self.num_numerical_leq + self.num_numerical_gr + self.num_ordinal_leq + \
                         self.num_ordinal_gr + self.num_nominal_eq + self.num_nominal_neq
        num_total_conditions = np.sum(num_conditions)

        if num_total_conditions > 0:
            frac_numerical_leq = np.sum(self.num_numerical_leq) / num_total_conditions * 100
            frac_numerical_gr = np.sum(self.num_numerical_gr) / num_total_conditions * 100
            frac_ordinal_leq = np.sum(self.num_ordinal_leq) / num_total_conditions * 100
            frac_ordinal_gr = np.sum(self.num_ordinal_gr) / num_total_conditions * 100
            frac_nominal_eq = np.sum(self.num_nominal_eq) / num_total_conditions * 100
            frac_nominal_neq = np.sum(self.num_nominal_neq) / num_total_conditions * 100
            num_conditions_mean = np.mean(num_conditions)
            num_conditions_min = np.min(num_conditions)
            num_conditions_max = np.max(num_conditions)
        else:
            frac_numerical_leq = 0.0
            frac_numerical_gr = 0.0
            frac_ordinal_leq = 0.0
            frac_ordinal_gr = 0.0
            frac_nominal_eq = 0.0
            frac_nominal_neq = 0.0
            num_conditions_mean = 0.0
            num_conditions_min = 0.0
            num_conditions_max = 0.0

        num_total_predictions = np.sum(num_predictions)

        if num_total_predictions > 0:
            frac_pos = np.sum(self.num_pos_predictions) / num_total_predictions * 100
            frac_neg = np.sum(self.num_neg_predictions) / num_total_predictions * 100
            num_predictions_mean = np.mean(num_predictions)
            num_predictions_min = np.min(num_predictions)
            num_predictions_max = np.max(num_predictions)
        else:
            frac_pos = 0.0
            frac_neg = 0.0
            num_predictions_mean = 0.0
            num_predictions_min = 0.0
            num_predictions_max = 0.0

        num_rules = num_predictions.shape[0]

        header = [
            'Statistics about conditions', 'Total', 'Numerical <= operator', 'Numerical > operator',
            'Ordinal <= operator', 'Ordinal > operator', 'Nominal == operator', 'Nominal != operator'
        ]
        alignment = ['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
        rows = []

        if self.default_rule_index is not None:
            rows.append([
                'Default rule',
                str(0),
                format_percentage(0),
                format_percentage(0),
                format_percentage(0),
                format_percentage(0),
                format_percentage(0),
                format_percentage(0)
            ])

        rows.append([
            str(num_rules) + ' local rules',
            str(num_total_conditions),
            format_percentage(frac_numerical_leq),
            format_percentage(frac_numerical_gr),
            format_percentage(frac_ordinal_leq),
            format_percentage(frac_ordinal_gr),
            format_percentage(frac_nominal_eq),
            format_percentage(frac_nominal_neq)
        ])
        text = format_table(rows, header=header, alignment=alignment) + '\n\n'

        header = ['Statistics about predictions', 'Total', 'Positive', 'Negative']
        alignment = ['left', 'right', 'right', 'right']
        rows = []

        if self.default_rule_index is not None:
            default_rule_num_predictions = self.default_rule_pos_predictions + self.default_rule_neg_predictions
            default_rule_frac_pos = self.default_rule_pos_predictions / default_rule_num_predictions * 100
            default_rule_frac_neg = self.default_rule_neg_predictions / default_rule_num_predictions * 100
            rows.append([
                'Default rule',
                str(default_rule_num_predictions),
                format_percentage(default_rule_frac_pos),
                format_percentage(default_rule_frac_neg)
            ])

        rows.append([
            str(num_rules) + ' local rules',
            str(num_total_predictions),
            format_percentage(frac_pos),
            format_percentage(frac_neg)
        ])
        text += format_table(rows, header=header, alignment=alignment) + '\n\n'

        header = ['Statistics per local rule', 'Minimum', 'Average', 'Maximum']
        rows = []
        rows.append([
            'Conditions',
            format_number(num_conditions_min),
            format_number(num_conditions_mean),
            format_number(num_conditions_max)
        ])
        rows.append([
            'Predictions',
            format_number(num_predictions_min),
            format_number(num_predictions_mean),
            format_number(num_predictions_max)
        ])
        return text + format_table(rows, header=header)

    # pylint: disable=unused-argument
    def to_table(self, options: Options, **_) -> Optional[TabularOutputData.Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        rows = []
        default_rule_index = self.default_rule_index
        num_rules = len(self.num_pos_predictions)
        num_total_rules = num_rules if default_rule_index is None else num_rules + 1
        j = 0

        for i in range(num_total_rules):
            rule_name = 'Rule ' + str(i + 1)

            if i == default_rule_index:
                rule_name += ' (Default rule)'
                num_numerical_leq = 0
                num_numerical_gr = 0
                num_ordinal_leq = 0
                num_ordinal_gr = 0
                num_nominal_eq = 0
                num_nominal_neq = 0
                num_pos_predictions = self.default_rule_pos_predictions
                num_neg_predictions = self.default_rule_neg_predictions
            else:
                num_numerical_leq = self.num_numerical_leq[j]
                num_numerical_gr = self.num_numerical_gr[j]
                num_ordinal_leq = self.num_ordinal_leq[j]
                num_ordinal_gr = self.num_ordinal_gr[j]
                num_nominal_eq = self.num_nominal_eq[j]
                num_nominal_neq = self.num_nominal_neq[j]
                num_pos_predictions = self.num_pos_predictions[j]
                num_neg_predictions = self.num_neg_predictions[j]
                j += 1

            num_numerical = num_numerical_leq + num_numerical_gr
            num_ordinal = num_ordinal_leq + num_ordinal_gr
            num_nominal = num_nominal_eq + num_nominal_neq
            num_conditions = num_numerical + num_ordinal + num_nominal
            num_predictions = num_pos_predictions + num_neg_predictions
            rows.append({
                'Rule': rule_name,
                'conditions': num_conditions,
                'numerical conditions': num_numerical,
                'numerical <= operator': num_numerical_leq,
                'numerical > operator': num_numerical_gr,
                'ordinal conditions': num_ordinal,
                'ordinal <= operator': num_ordinal_leq,
                'ordinal > operator': num_ordinal_gr,
                'nominal conditions': num_nominal,
                'nominal == operator': num_nominal_eq,
                'nominal != operator': num_nominal_neq,
                'predictions': num_predictions,
                'pos. predictions': num_pos_predictions,
                'neg. predictions': num_neg_predictions
            })

        return rows
