#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for printing textual representations of models. The models can be written to one or several outputs,
e.g. to the console or to a file.
"""
import logging as log
from _io import StringIO
from abc import ABC, abstractmethod
from typing import List, Set

import numpy as np
from mlrl.common.cython.model import RuleModelVisitor, EmptyBody, ConjunctiveBody, CompleteHead, PartialHead

from mlrl.common.learners import Learner
from mlrl.common.options import Options
from mlrl.testbed.data import Attribute, MetaData
from mlrl.testbed.io import clear_directory, open_writable_txt_file

ARGUMENT_PRINT_FEATURE_NAMES = 'print_feature_names'

ARGUMENT_PRINT_LABEL_NAMES = 'print_label_names'

ARGUMENT_PRINT_NOMINAL_VALUES = 'print_nominal_values'

PRINT_OPTION_VALUES: Set[str] = {ARGUMENT_PRINT_FEATURE_NAMES, ARGUMENT_PRINT_LABEL_NAMES,
                                 ARGUMENT_PRINT_NOMINAL_VALUES}


class RuleModelFormatter(RuleModelVisitor):
    """
    Allows to create textual representation of the rules in a `RuleModel`.
    """

    def __init__(self, attributes: List[Attribute], labels: List[Attribute], print_feature_names: bool,
                 print_label_names: bool, print_nominal_values: bool):
        """
        :param attributes:              A list that contains the attributes
        :param labels:                  A list that contains the labels
        :param print_feature_names:     True, if the names of features should be printed, False otherwise
        :param print_label_names:       True, if the names of labels should be printed, False otherwise
        :param print_nominal_values:    True, if the values of nominal values should be printed, False otherwise
        """
        self.print_feature_names = print_feature_names
        self.print_label_names = print_label_names
        self.print_nominal_values = print_nominal_values
        self.attributes = attributes
        self.labels = labels
        self.text = StringIO()

    def visit_empty_body(self, _: EmptyBody):
        self.text.write('{}')

    def __format_conditions(self, num_conditions: int, indices: np.ndarray, thresholds: np.ndarray,
                            operator: str) -> int:
        result = num_conditions

        if indices is not None and thresholds is not None:
            text = self.text
            attributes = self.attributes
            print_feature_names = self.print_feature_names
            print_nominal_values = self.print_nominal_values

            for i in range(indices.shape[0]):
                if result > 0:
                    text.write(' & ')

                feature_index = indices[i]
                threshold = thresholds[i]
                attribute = attributes[feature_index] if len(attributes) > feature_index else None

                if print_feature_names and attribute is not None:
                    text.write(attribute.attribute_name)
                else:
                    text.write(str(feature_index))

                text.write(' ')
                text.write(operator)
                text.write(' ')

                if attribute is not None and attribute.nominal_values is not None:
                    if print_nominal_values and len(attribute.nominal_values) > threshold:
                        text.write('"' + attribute.nominal_values[threshold] + '"')
                    else:
                        text.write(str(threshold))
                else:
                    text.write(str(threshold))

                result += 1

        return result

    def visit_conjunctive_body(self, body: ConjunctiveBody):
        text = self.text
        text.write('{')
        num_conditions = self.__format_conditions(0, body.leq_indices, body.leq_thresholds, '<=')
        num_conditions = self.__format_conditions(num_conditions, body.gr_indices, body.gr_thresholds, '>')
        num_conditions = self.__format_conditions(num_conditions, body.eq_indices, body.eq_thresholds, '==')
        self.__format_conditions(num_conditions, body.neq_indices, body.neq_thresholds, '!=')
        text.write('}')

    def visit_complete_head(self, head: CompleteHead):
        text = self.text
        print_label_names = self.print_label_names
        labels = self.labels
        scores = head.scores
        text.write(' => (')

        for i in range(scores.shape[0]):
            if i > 0:
                text.write(', ')

            if print_label_names and len(labels) > i:
                text.write(labels[i].attribute_name)
            else:
                text.write(str(i))

            text.write(' = ')
            text.write('{0:.2f}'.format(scores[i]))

        text.write(')\n')

    def visit_partial_head(self, head: PartialHead):
        text = self.text
        print_label_names = self.print_label_names
        labels = self.labels
        indices = head.indices
        scores = head.scores
        text.write(' => (')

        for i in range(indices.shape[0]):
            if i > 0:
                text.write(', ')

            label_index = indices[i]

            if print_label_names and len(labels) > label_index:
                text.write(labels[label_index].attribute_name)
            else:
                text.write(str(label_index))

            text.write(' = ')
            text.write('{0:.2f}'.format(scores[i]))

        text.write(')\n')

    def get_text(self) -> str:
        """
        Returns the textual representation that has been created via the `format` method.

        :return: The textual representation
        """
        return self.text.getvalue()


class ModelPrinterOutput(ABC):
    """
    An abstract base class for all outputs, textual representations of models may be written to.
    """

    @abstractmethod
    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        """
        Write a textual representation of a model to the output.

        :param experiment_name:     The name of the experiment
        :param model:               The textual representation of the model
        :param total_folds:         The total number of folds
        :param fold:                The fold for which the results should be written or None, if no cross validation is
                                    used or if the overall results, averaged over all folds, should be written
        """
        pass


class ModelPrinter(ABC):
    """
    An abstract base class for all classes that allow to print a textual representation of a `MLLearner`'s model.
    """

    def __init__(self, print_options: str, outputs: List[ModelPrinterOutput]):
        """
        :param print_options:   The options to be used for printing models
        :param outputs:         The outputs, the textual representations of models should be written to
        """
        self.outputs = outputs

        try:
            self.print_options = Options.create(print_options, PRINT_OPTION_VALUES)
        except ValueError as e:
            raise ValueError('Invalid value given for parameter "print_options". ' + str(e))

    def print(self, experiment_name: str, meta_data: MetaData, learner: Learner, current_fold: int, num_folds: int):
        """
        Prints a textual representation of a `MLLearner`'s model.

        :param experiment_name: The name of the experiment
        :param meta_data:       The meta data of the training data set
        :param learner:         The learner
        :param current_fold:    The current cross validation fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        model = learner.model_
        text = self._format_model(meta_data, model)

        for output in self.outputs:
            output.write_model(experiment_name, text, num_folds, current_fold if num_folds > 1 else None)

    @abstractmethod
    def _format_model(self, meta_data: MetaData, model) -> str:
        """
        Must be implemented by subclasses in order to create a textual representation of a model.

        :param meta_data:   The meta data of the training data set
        :param model:       The model
        :return:            The textual representation of the given model
        """
        pass


class ModelPrinterLogOutput(ModelPrinterOutput):
    """
    Outputs the textual representation of a model using the logger.
    """

    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        msg = 'Model for experiment \"' + experiment_name + '\"' + (
            ' (Fold ' + str(fold + 1) + ')' if fold is not None else '') + ':\n\n%s'
        log.info(msg, model)


class ModelPrinterTxtOutput(ModelPrinterOutput):
    """
    Writes the textual representation of a model to a text file.
    """

    def __init__(self, output_dir: str, clear_dir: bool = True):
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        with open_writable_txt_file(self.output_dir, 'rules', fold, append=False) as text_file:
            text_file.write(model)

    def __clear_dir_if_necessary(self):
        """
        Clears the output directory, if necessary.
        """
        if self.clear_dir:
            clear_directory(self.output_dir)
            self.clear_dir = False


class RulePrinter(ModelPrinter):
    """
    Allows to print a textual representation of a `MLRuleLearner`'s rule-based model.
    """

    def __init__(self, print_options: str, outputs: List[ModelPrinterOutput]):
        super().__init__(print_options, outputs)

    def _format_model(self, meta_data: MetaData, model) -> str:
        print_options = self.print_options
        print_feature_names = print_options.get_bool(ARGUMENT_PRINT_FEATURE_NAMES, True)
        print_label_names = print_options.get_bool(ARGUMENT_PRINT_LABEL_NAMES, True)
        print_nominal_values = print_options.get_bool(ARGUMENT_PRINT_NOMINAL_VALUES, True)
        formatter = RuleModelFormatter(attributes=meta_data.attributes, labels=meta_data.labels,
                                       print_feature_names=print_feature_names, print_label_names=print_label_names,
                                       print_nominal_values=print_nominal_values)
        model.visit(formatter)
        return formatter.get_text()
