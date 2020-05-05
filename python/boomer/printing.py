#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for printing textual representations of models. The models can be written to one or several outputs,
e.g. to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod

import numpy as np
from boomer.algorithm._model import Rule, Body, EmptyBody, ConjunctiveBody, Head, FullHead, PartialHead

from boomer.algorithm.model import Theory
from boomer.algorithm.rule_learners import MLRuleLearner
from boomer.learners import MLLearner
from boomer.stats import Stats


class ModelPrinter(ABC):
    """
    An abstract base class for all classes that allow to print a textual representation of a `MLLearner`'s model.
    """

    @abstractmethod
    def print(self, experiment_name: str, learner: MLLearner, first_fold: int, current_fold: int, last_fold: int,
              num_folds: int):
        """
        Prints a textual representation of a `MLLearner`'s model.

        :param experiment_name: The name of the experiment
        :param learner:         The learner
        :param first_fold:      The first cross validation fold or 0, if no cross validation is used
        :param current_fold:    The current cross validation fold starting at 0, or 0 if no cross validation is used
        :param last_fold:       The last cross validation fold or 0, if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        pass


class ModelPrinterOutput(ABC):
    """
    An abstract base class for all outputs, textual representations of models may be written to.
    """

    @abstractmethod
    def write_model(self, experiment_name: str, model: str, current_fold: int, num_folds: int):
        """
        Write a textual representation of a model to the output.

        :param experiment_name: The name of the experiment
        :param model:           The textual representation of the model
        :param current_fold:    The current cross validation fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        pass


class ModelPrinterLogOutput(ModelPrinterOutput):
    """
    Outputs the textual representation of a model using the logger.
    """

    def write_model(self, experiment_name: str, model: str, current_fold: int, num_folds: int):
        msg = 'Model for experiment \"' + experiment_name + '\"' + (
            ' (Fold ' + str(current_fold + 1) + ')' if num_folds > 1 else '') + ':\n\n%s\n'
        log.info(msg, model)


class RulePrinter(ModelPrinter):
    """
    Allows to print a textual representation of a `MLRuleLearner`'s rule-based model.
    """

    def __init__(self, *args: ModelPrinterOutput):
        """
        :param args: The outputs, the textual representations of rules should be written to
        """
        self.outputs = args

    def print(self, experiment_name: str, learner: MLLearner, first_fold: int, current_fold: int, last_fold: int,
              num_folds: int):
        if isinstance(learner, MLRuleLearner):
            stats = learner.stats_
            theory = learner.model_
            formatted_theory = format_theory(stats, theory)

            for output in self.outputs:
                output.write_model(experiment_name, formatted_theory, current_fold, num_folds)
        else:
            raise ValueError('Unsupported model type: ' + type(learner).__name__)


def format_theory(stats: Stats, theory: Theory) -> str:
    """
    Formats a specific theory as a text.

    :param stats:   Statistics about the training data set
    :param theory:  The theory to be formatted
    :return:        The text
    """
    text = ''

    for rule in theory:
        if len(text) > 0:
            text += '\n'

        text += format_rule(stats, rule)

    return text


def format_rule(stats: Stats, rule: Rule) -> str:
    """
    Formats a specific rule as a text.

    :param stats:   Statistics about the training data set
    :param rule:    The rule to be formatted
    :return:        The text
    """
    text = __format_body(rule.body)
    text += ' -> '
    text += __format_head(stats, rule.head)
    return text


def __format_body(body: Body) -> str:
    """
    Formats the body of a rule as a text.

    :param body:    The body to be formatted
    :return:        The text
    """
    if isinstance(body, EmptyBody):
        return '{}'
    elif isinstance(body, ConjunctiveBody):
        return '{' + __format_conjunctive_body(body) + '}'
    else:
        raise ValueError('Body has unknown type: ' + type(body).__name__)


def __format_conjunctive_body(body: ConjunctiveBody) -> str:
    """
    Formats the conjunctive body of a rule as a text.

    :param body:    The conjunctive body to be formatted
    :return:        The text
    """
    text = ''

    if body.leq_feature_indices is not None and body.leq_thresholds is not None:
        text = __format_conditions(np.asarray(body.leq_feature_indices), np.asarray(body.leq_thresholds), '<=', text)

    if body.gr_feature_indices is not None and body.gr_thresholds is not None:
        text = __format_conditions(np.asarray(body.gr_feature_indices), np.asarray(body.gr_thresholds), '>', text)

    if body.eq_feature_indices is not None and body.eq_thresholds is not None:
        text = __format_conditions(np.asarray(body.eq_feature_indices), np.asarray(body.eq_thresholds), '==', text)

    if body.neq_feature_indices is not None and body.neq_thresholds is not None:
        text = __format_conditions(np.asarray(body.neq_feature_indices), np.asarray(body.neq_thresholds), '!=', text)

    return text


def __format_conditions(feature_indices: np.ndarray, thresholds: np.ndarray, operator: str, text: str) -> str:
    """
    Formats conditions that are contained by the body of a rule and the textual representation to an existing text.

    :param feature_indices: An array of dtype int, shape `(num_conditions)`, representing the feature indices that
                            correspond to the conditions
    :param thresholds:      An array of dtype float, shape `(num_conditions)`, representing the thresholds used by the
                            conditions
    :param operator:        A textual representation of the operator that is used by the conditions
    :param text:            The text, the textual representation of the conditions should be appended to
    :return:                The given text including the appended text
    """
    for i in range(feature_indices.shape[0]):
        if len(text) > 0:
            text += ' & '

        text += str(feature_indices[i])
        text += ' '
        text += operator
        text += ' '
        text += str(thresholds[i])

    return text


def __format_head(stats: Stats, head: Head) -> str:
    """
    Formats the head of a rule as a text.

    :param stats:   Statistics about the training data set
    :param head:    The head to be formatted
    :return:        The text
    """
    if isinstance(head, FullHead):
        return '(' + __format_full_head(head) + ')'
    elif isinstance(head, PartialHead):
        return '(' + __format_partial_head(stats, head) + ')'
    else:
        raise ValueError('Head has unknown type: ' + type(head).__name__)


def __format_full_head(head: FullHead) -> str:
    """
    Formats the full head of a rule as a text.

    :param head:    The full head to be formatted
    :return:        The text
    """
    text = ''
    scores = np.asarray(head.scores)

    for i in range(scores.shape[0]):
        if len(text) > 0:
            text += ', '

        text += '{0:.2f}'.format(scores[i])

    return text


def __format_partial_head(stats: Stats, head: PartialHead) -> str:
    """
    Formats the partial head of a rule as a text.

    :param stats:   Statistics about the training data set
    :param head:    The partial head to be formatted
    :return:        The text
    """
    text = ''
    scores = np.asarray(head.scores)
    label_indices = np.asarray(head.label_indices)

    for i in range(stats.num_labels):
        if len(text) > 0:
            text += ', '

        label_index = np.argwhere(label_indices == i)

        if np.size(label_index) > 0:
            text += '{0:.2f}'.format(scores[label_index.item()])
        else:
            text += '?'

    return text
