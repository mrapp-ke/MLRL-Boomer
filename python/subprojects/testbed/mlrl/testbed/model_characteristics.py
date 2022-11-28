"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of models. The characteristics can be written to one or several
outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod

import numpy as np
from mlrl.common.cython.rule_model import RuleModel, RuleModelVisitor, EmptyBody, ConjunctiveBody, CompleteHead, \
    PartialHead
from mlrl.common.learners import Learner
from mlrl.testbed.data_splitting import DataSplit
from mlrl.testbed.format import format_table, format_percentage, format_float
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer
from typing import List


class RuleModelCharacteristics:
    """
    Stores the characteristics of a `RuleModel`.
    """

    def __init__(self, default_rule_index: int, default_rule_pos_predictions: int, default_rule_neg_predictions: int,
                 num_leq: np.ndarray, num_gr: np.ndarray, num_eq: np.ndarray, num_neq: np.ndarray,
                 num_pos_predictions: np.ndarray, num_neg_predictions: np.ndarray):
        """
        :param default_rule_index:              The index of the default rule or None, if no default rule is used
        :param default_rule_pos_predictions:    The number of positive predictions of the default rule, if any
        :param default_rule_neg_predictions:    The number of negative predictions of the default rule, if any
        :param num_leq:                         A `np.ndarray`, shape `(num_rules)` that stores the number of conditions
                                                that use the <= operator per rule
        :param num_gr:                          A `np.ndarray`, shape `(num_rules)` that stores the number of conditions
                                                that use the > operator per rule
        :param num_eq:                          A `np.ndarray`, shape `(num_rules)` that stores the number of conditions
                                                that use the == operator per rule
        :param num_neq:                         A `np.ndarray`, shape `(num_rules)` that stores the number of conditions
                                                that use the != operator per rule
        :param num_pos_predictions:             A `np.ndarray`, shape `(num_rules)` that stores the number of positive
                                                predictions per rule
        :param num_neg_predictions:             A `np.ndarray`, shape `(num_rules)` that stores the number of negative
                                                predictions per rule
        """
        self.default_rule_index = default_rule_index
        self.default_rule_pos_predictions = default_rule_pos_predictions
        self.default_rule_neg_predictions = default_rule_neg_predictions
        self.num_leq = num_leq
        self.num_gr = num_gr
        self.num_eq = num_eq
        self.num_neq = num_neq
        self.num_pos_predictions = num_pos_predictions
        self.num_neg_predictions = num_neg_predictions


class RuleModelCharacteristicsOutput(ABC):
    """
    An abstract base class for all outputs, the characteristics of a rule-based model may be written to.
    """

    @abstractmethod
    def write_model_characteristics(self, data_split: DataSplit, characteristics: RuleModelCharacteristics):
        """
        Writes the characteristics of a rule-based model to the output.

        :param characteristics: The characteristics of the model
        :param data_split:      The split of the available data, the characteristics correspond to
        """
        pass


class RuleModelCharacteristicsVisitor(RuleModelVisitor):
    """
    A visitor that allows to determine the characteristics of a `RuleModel`.
    """

    def __init__(self):
        self.num_leq = []
        self.num_gr = []
        self.num_eq = []
        self.num_neq = []
        self.num_pos_predictions = []
        self.num_neg_predictions = []
        self.default_rule_index = None
        self.default_rule_pos_predictions = 0
        self.default_rule_neg_predictions = 0
        self.index = -1

    def visit_empty_body(self, _: EmptyBody):
        self.index += 1
        self.default_rule_index = self.index

    def visit_conjunctive_body(self, body: ConjunctiveBody):
        self.index += 1
        self.num_leq.append(body.leq_indices.shape[0] if body.leq_indices is not None else 0)
        self.num_gr.append(body.gr_indices.shape[0] if body.gr_indices is not None else 0)
        self.num_eq.append(body.eq_indices.shape[0] if body.eq_indices is not None else 0)
        self.num_neq.append(body.neq_indices.shape[0] if body.neq_indices is not None else 0)

    def visit_complete_head(self, head: CompleteHead):
        num_pos_predictions = np.count_nonzero(head.scores > 0)
        num_neg_predictions = head.scores.shape[0] - num_pos_predictions

        if self.index == self.default_rule_index:
            self.default_rule_pos_predictions = num_pos_predictions
            self.default_rule_neg_predictions = num_neg_predictions
        else:
            self.num_pos_predictions.append(num_pos_predictions)
            self.num_neg_predictions.append(num_neg_predictions)

    def visit_partial_head(self, head: PartialHead):
        num_pos_predictions = np.count_nonzero(head.scores > 0)
        num_neg_predictions = head.scores.shape[0] - num_pos_predictions

        if self.index == self.default_rule_index:
            self.default_rule_pos_predictions = num_pos_predictions
            self.default_rule_neg_predictions = num_neg_predictions
        else:
            self.num_pos_predictions.append(num_pos_predictions)
            self.num_neg_predictions.append(num_neg_predictions)


class RuleModelCharacteristicsLogOutput(RuleModelCharacteristicsOutput):
    """
    Outputs the characteristics of a `RuleModel` using the logger.
    """

    def write_model_characteristics(self, data_split: DataSplit, characteristics: RuleModelCharacteristics):
        default_rule_index = characteristics.default_rule_index
        num_pos_predictions = characteristics.num_pos_predictions
        num_neg_predictions = characteristics.num_neg_predictions
        num_predictions = num_pos_predictions + num_neg_predictions
        num_leq = characteristics.num_leq
        num_gr = characteristics.num_gr
        num_eq = characteristics.num_eq
        num_neq = characteristics.num_neq
        num_conditions = num_leq + num_gr + num_eq + num_neq
        num_total_conditions = np.sum(num_conditions)

        if num_total_conditions > 0:
            frac_leq = np.sum(num_leq) / num_total_conditions * 100
            frac_gr = np.sum(num_gr) / num_total_conditions * 100
            frac_eq = np.sum(num_eq) / num_total_conditions * 100
            frac_neq = np.sum(num_neq) / num_total_conditions * 100
            num_conditions_mean = np.mean(num_conditions)
            num_conditions_min = np.min(num_conditions)
            num_conditions_max = np.max(num_conditions)

        else:
            frac_leq = 0.0
            frac_gr = 0.0
            frac_eq = 0.0
            frac_neq = 0.0
            num_conditions_mean = 0.0
            num_conditions_min = 0.0
            num_conditions_max = 0.0

        num_total_predictions = np.sum(num_predictions)

        if num_total_predictions > 0:
            frac_pos = np.sum(num_pos_predictions) / num_total_predictions * 100
            frac_neg = np.sum(num_neg_predictions) / num_total_predictions * 100
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
        msg = 'Model characteristics'

        if data_split.is_cross_validation_used():
            msg += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

        msg += ':\n\n'
        header = ['Statistics about conditions', 'Total', '<= operator', '> operator', '== operator', '!= operator']
        alignment = ['left', 'right', 'right', 'right', 'right', 'right']
        rows = []

        if default_rule_index is not None:
            rows.append(['Default rule', str(0), format_percentage(0), format_percentage(0), format_percentage(0),
                         format_percentage(0)])

        rows.append([str(num_rules) + ' local rules', str(num_total_conditions), format_percentage(frac_leq),
                     format_percentage(frac_gr), format_percentage(frac_eq), format_percentage(frac_neq)])
        msg += format_table(rows, header=header, alignment=alignment) + '\n\n'

        header = ['Statistics about predictions', 'Total', 'Positive', 'Negative']
        alignment = ['left', 'right', 'right', 'right']
        rows = []

        if default_rule_index is not None:
            default_rule_num_predictions = characteristics.default_rule_pos_predictions \
                                           + characteristics.default_rule_neg_predictions
            default_rule_frac_pos = characteristics.default_rule_pos_predictions / default_rule_num_predictions * 100
            default_rule_frac_neg = characteristics.default_rule_neg_predictions / default_rule_num_predictions * 100
            rows.append(
                ['Default rule', str(default_rule_num_predictions), format_percentage(default_rule_frac_pos),
                 format_percentage(default_rule_frac_neg)])

        rows.append([str(num_rules) + ' local rules', str(num_total_predictions), format_percentage(frac_pos),
                     format_percentage(frac_neg)])
        msg += format_table(rows, header=header, alignment=alignment) + '\n\n'

        header = ['Statistics per local rule', 'Minimum', 'Average', 'Maximum']
        rows = [['Conditions', format_float(num_conditions_min), format_float(num_conditions_mean),
                 format_float(num_conditions_max)],
                ['Predictions', format_float(num_predictions_min), format_float(num_predictions_mean),
                 format_float(num_predictions_max)]]
        msg += format_table(rows, header=header) + '\n\n'
        log.info(msg)


class RuleModelCharacteristicsCsvOutput(RuleModelCharacteristicsOutput):
    """
    Writes the characteristics of a `RuleModel` to a CSV file.
    """

    COL_RULE_NAME = 'Rule'

    COL_CONDITIONS = 'conditions'

    COL_NUMERICAL_CONDITIONS = 'numerical conditions'

    COL_LEQ_CONDITIONS = 'conditions using <= operator'

    COL_GR_CONDITIONS = 'conditions using > operator'

    COL_NOMINAL_CONDITIONS = 'nominal conditions'

    COL_EQ_CONDITIONS = 'conditions using == operator'

    COL_NEQ_CONDITIONS = 'conditions using != operator'

    COL_PREDICTIONS = 'predictions'

    COL_POS_PREDICTIONS = 'pos. predictions'

    COL_NEG_PREDICTIONS = 'neg. predictions'

    def __init__(self, output_dir: str):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir

    def write_model_characteristics(self, data_split: DataSplit, characteristics: RuleModelCharacteristics):
        header = [
            RuleModelCharacteristicsCsvOutput.COL_RULE_NAME,
            RuleModelCharacteristicsCsvOutput.COL_CONDITIONS,
            RuleModelCharacteristicsCsvOutput.COL_NUMERICAL_CONDITIONS,
            RuleModelCharacteristicsCsvOutput.COL_LEQ_CONDITIONS,
            RuleModelCharacteristicsCsvOutput.COL_GR_CONDITIONS,
            RuleModelCharacteristicsCsvOutput.COL_NOMINAL_CONDITIONS,
            RuleModelCharacteristicsCsvOutput.COL_EQ_CONDITIONS,
            RuleModelCharacteristicsCsvOutput.COL_NEQ_CONDITIONS,
            RuleModelCharacteristicsCsvOutput.COL_PREDICTIONS,
            RuleModelCharacteristicsCsvOutput.COL_POS_PREDICTIONS,
            RuleModelCharacteristicsCsvOutput.COL_NEG_PREDICTIONS
        ]
        default_rule_index = characteristics.default_rule_index
        num_rules = len(characteristics.num_pos_predictions)
        num_total_rules = num_rules if default_rule_index is None else num_rules + 1

        with open_writable_csv_file(self.output_dir, 'model_characteristics', data_split.get_fold()) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            n = 0

            for i in range(num_total_rules):
                rule_name = 'Rule ' + str(i + 1)

                if i == default_rule_index:
                    rule_name += ' (Default rule)'
                    num_leq = 0
                    num_gr = 0
                    num_eq = 0
                    num_neq = 0
                    num_pos_predictions = characteristics.default_rule_pos_predictions
                    num_neg_predictions = characteristics.default_rule_neg_predictions
                else:
                    num_leq = characteristics.num_leq[n]
                    num_gr = characteristics.num_gr[n]
                    num_eq = characteristics.num_eq[n]
                    num_neq = characteristics.num_neq[n]
                    num_pos_predictions = characteristics.num_pos_predictions[n]
                    num_neg_predictions = characteristics.num_neg_predictions[n]
                    n += 1

                num_numerical = num_leq + num_gr
                num_nominal = num_eq + num_neq
                num_conditions = num_numerical + num_nominal
                num_predictions = num_pos_predictions + num_neg_predictions
                columns = {
                    RuleModelCharacteristicsCsvOutput.COL_RULE_NAME: rule_name,
                    RuleModelCharacteristicsCsvOutput.COL_CONDITIONS: num_conditions,
                    RuleModelCharacteristicsCsvOutput.COL_NUMERICAL_CONDITIONS: num_numerical,
                    RuleModelCharacteristicsCsvOutput.COL_LEQ_CONDITIONS: num_leq,
                    RuleModelCharacteristicsCsvOutput.COL_GR_CONDITIONS: num_gr,
                    RuleModelCharacteristicsCsvOutput.COL_NOMINAL_CONDITIONS: num_nominal,
                    RuleModelCharacteristicsCsvOutput.COL_EQ_CONDITIONS: num_eq,
                    RuleModelCharacteristicsCsvOutput.COL_NEQ_CONDITIONS: num_neq,
                    RuleModelCharacteristicsCsvOutput.COL_PREDICTIONS: num_predictions,
                    RuleModelCharacteristicsCsvOutput.COL_POS_PREDICTIONS: num_pos_predictions,
                    RuleModelCharacteristicsCsvOutput.COL_NEG_PREDICTIONS: num_neg_predictions
                }
                csv_writer.writerow(columns)


class ModelCharacteristicsPrinter(ABC):
    """
    A class that allows to print the characteristics of a learner's model.
    """

    def print(self, data_split: DataSplit, learner):
        """
        Prints the characteristics of a learner's model. If the learner does not support to obtain the characteristics
        of the model, a `ValueError` is raised.

        :param data_split:  The split of the available data, the model corresponds to
        :param learner:     The learner
        """
        if not isinstance(learner, Learner):
            raise ValueError('Cannot obtain characteristics of a model of type ' + type(learner.__name__))

        self._print_model_characteristics(data_split, learner.model_)

    def _print_model_characteristics(self, data_split: DataSplit, model):
        """
        :param data_split:  The split of the available data, the model corresponds to
        :param model:       The model
        """
        pass


class RuleModelCharacteristicsPrinter(ModelCharacteristicsPrinter):
    """
    A class that allows to print the characteristics of a rule-based model.
    """

    def __init__(self, outputs: List[RuleModelCharacteristicsOutput]):
        """
        :param outputs: The outputs, the model characteristics should be written to
        """
        self.outputs = outputs

    def _print_model_characteristics(self, data_split: DataSplit, model):
        if not isinstance(model, RuleModel):
            raise ValueError('Cannot obtain characteristics of a model of type ' + type(model).__name__)

        if len(self.outputs) > 0:
            visitor = RuleModelCharacteristicsVisitor()
            model.visit_used(visitor)
            characteristics = RuleModelCharacteristics(
                default_rule_index=visitor.default_rule_index,
                default_rule_pos_predictions=visitor.default_rule_pos_predictions,
                default_rule_neg_predictions=visitor.default_rule_neg_predictions,
                num_leq=np.asarray(visitor.num_leq),
                num_gr=np.asarray(visitor.num_gr),
                num_eq=np.asarray(visitor.num_eq),
                num_neq=np.asarray(visitor.num_neq),
                num_pos_predictions=np.asarray(visitor.num_pos_predictions),
                num_neg_predictions=np.asarray(visitor.num_neg_predictions))

            for output in self.outputs:
                output.write_model_characteristics(data_split, characteristics)
