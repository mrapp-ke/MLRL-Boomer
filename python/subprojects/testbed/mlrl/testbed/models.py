"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing textual representations of models. The models can be written to one or several outputs,
e.g., to the console or to a file.
"""
import logging as log
from _io import StringIO
from abc import ABC, abstractmethod

import numpy as np
from mlrl.common.cython.rule_model import RuleModel, RuleModelVisitor, EmptyBody, ConjunctiveBody, CompleteHead, \
    PartialHead
from mlrl.common.learners import Learner
from mlrl.common.options import Options
from mlrl.testbed.data import Attribute, MetaData
from mlrl.testbed.data_splitting import DataSplit
from mlrl.testbed.io import open_writable_txt_file
from typing import List, Optional

OPTION_PRINT_FEATURE_NAMES = 'print_feature_names'

OPTION_PRINT_LABEL_NAMES = 'print_label_names'

OPTION_PRINT_NOMINAL_VALUES = 'print_nominal_values'

OPTION_PRINT_BODIES = 'print_bodies'

OPTION_PRINT_HEADS = 'print_heads'


class RuleModelFormatter(RuleModelVisitor):
    """
    Allows to create textual representations of the rules in a `RuleModel`.
    """

    def __init__(self, options: Options, meta_data: MetaData):
        """
        :param options:     The options that should be used for creating textual representations of the rules in a model
        :param meta_data:   The meta-data of the training data set
        """

        self.print_feature_names = options.get_bool(OPTION_PRINT_FEATURE_NAMES, True)
        self.print_label_names = options.get_bool(OPTION_PRINT_LABEL_NAMES, True)
        self.print_nominal_values = options.get_bool(OPTION_PRINT_NOMINAL_VALUES, True)
        self.print_bodies = options.get_bool(OPTION_PRINT_BODIES, True)
        self.print_heads = options.get_bool(OPTION_PRINT_HEADS, True)
        self.attributes = meta_data.attributes
        self.labels = meta_data.labels
        self.text = StringIO()

    def visit_empty_body(self, _: EmptyBody):
        if self.print_bodies:
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
                attribute: Optional[Attribute] = attributes[feature_index] if len(attributes) > feature_index else None

                if print_feature_names and attribute is not None:
                    text.write(attribute.attribute_name)
                else:
                    text.write(str(feature_index))

                text.write(' ')
                text.write(operator)
                text.write(' ')

                if attribute is not None and attribute.nominal_values is not None:
                    nominal_value = int(threshold)

                    if print_nominal_values and len(attribute.nominal_values) > nominal_value:
                        text.write('"' + attribute.nominal_values[nominal_value] + '"')
                    else:
                        text.write(str(nominal_value))
                else:
                    text.write(str(threshold))

                result += 1

        return result

    def visit_conjunctive_body(self, body: ConjunctiveBody):
        if self.print_bodies:
            text = self.text
            text.write('{')
            num_conditions = self.__format_conditions(0, body.leq_indices, body.leq_thresholds, '<=')
            num_conditions = self.__format_conditions(num_conditions, body.gr_indices, body.gr_thresholds, '>')
            num_conditions = self.__format_conditions(num_conditions, body.eq_indices, body.eq_thresholds, '==')
            self.__format_conditions(num_conditions, body.neq_indices, body.neq_thresholds, '!=')
            text.write('}')

    def visit_complete_head(self, head: CompleteHead):
        text = self.text

        if self.print_heads:
            print_label_names = self.print_label_names
            labels = self.labels
            scores = head.scores

            if self.print_bodies:
                text.write(' => ')

            text.write('(')

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
        elif self.print_bodies:
            text.write('\n')

    def visit_partial_head(self, head: PartialHead):
        text = self.text

        if self.print_heads:
            print_label_names = self.print_label_names
            labels = self.labels
            indices = head.indices
            scores = head.scores

            if self.print_bodies:
                text.write(' => ')

            text.write('(')

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
        elif self.print_bodies:
            text.write('\n')

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
    def get_options(self) -> Options:
        """
        Returns the options that should be used for creating textual representations of models.

        :return: The options that should be used
        """
        pass

    @abstractmethod
    def write_model(self, data_split: DataSplit, model: str):
        """
        Write a textual representation of a model to the output.

        :param data_split:  The split of the available data, the model corresponds to
        :param model:       The textual representation of the model
        """
        pass


class ModelPrinter(ABC):
    """
    An abstract base class for all classes that allow to print a textual representation of a learner's model.
    """

    def __init__(self, outputs: List[ModelPrinterOutput]):
        """
        :param outputs: The outputs, the textual representations of models should be written to
        """
        self.outputs = outputs

    def print(self, meta_data: MetaData, data_split: DataSplit, learner):
        """
        Prints a textual representation of a learner's model. If the learner does not support to create a textual
        representation of the model, a `ValueError` is raised.

        :param meta_data:   The meta-data of the training data set
        :param data_split:  The split of the available data, the model corresponds to
        :param learner:     The learner
        """
        if not isinstance(learner, Learner):
            raise ValueError('Cannot create textual representation of a model of type ' + type(learner).__name__)

        model = learner.model_

        for output in self.outputs:
            options = output.get_options()
            text = self._format_model(options, meta_data, model)
            output.write_model(data_split, text)

    @abstractmethod
    def _format_model(self, options: Options, meta_data: MetaData, model) -> str:
        """
        Must be implemented by subclasses in order to create a textual representation of a model.

        :param options:     The options that should be used for creating a textual representation of a model
        :param meta_data:   The meta-data of the training data set
        :param model:       The model
        :return:            The textual representation of the given model
        """
        pass


class ModelPrinterLogOutput(ModelPrinterOutput):
    """
    Outputs the textual representation of a model using the logger.
    """

    def __init__(self, options: Options):
        """
        :param options: The options to be used for printing models
        """
        self.options = options

    def get_options(self) -> Options:
        return self.options

    def write_model(self, data_split: DataSplit, model: str):
        msg = 'Model'

        if data_split.is_cross_validation_used():
            msg += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

        msg += ':\n\n%s'
        log.info(msg, model)


class ModelPrinterTxtOutput(ModelPrinterOutput):
    """
    Writes the textual representation of a model to a text file.
    """

    def __init__(self, options: Options, output_dir: str):
        """
        :param options:     The options to be used for printing models
        :param output_dir:  The path of the directory, the text files should be written to
        """
        self.options = options
        self.output_dir = output_dir

    def get_options(self) -> Options:
        return self.options

    def write_model(self, data_split: DataSplit, model: str):
        with open_writable_txt_file(self.output_dir, 'rules', data_split.get_fold()) as text_file:
            text_file.write(model)


class RulePrinter(ModelPrinter):
    """
    Allows to print a textual representation of a rule-based model.
    """

    def __init__(self, outputs: List[ModelPrinterOutput]):
        super(RulePrinter, self).__init__(outputs)

    def _format_model(self, options: Options, meta_data: MetaData, model) -> str:
        if not isinstance(model, RuleModel):
            raise ValueError('Cannot create a textual representation of a model of type ' + type(model).__name__)

        formatter = RuleModelFormatter(options, meta_data)
        model.visit_used(formatter)
        return formatter.get_text()
