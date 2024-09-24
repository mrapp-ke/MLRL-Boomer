"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing textual representations of models. The models can be written to one or several outputs,
e.g., to the console or to a file.
"""
import logging as log

from abc import ABC
from typing import Any, List, Optional

import numpy as np

from _io import StringIO

from mlrl.common.cython.rule_model import CompleteHead, ConjunctiveBody, EmptyBody, PartialHead, RuleModel, \
    RuleModelVisitor
from mlrl.common.mixins import ClassifierMixin, RegressorMixin
from mlrl.common.options import Options

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import format_float
from mlrl.testbed.output_writer import Formattable, OutputWriter
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType
from mlrl.testbed.problem_type import ProblemType

OPTION_PRINT_FEATURE_NAMES = 'print_feature_names'

OPTION_PRINT_OUTPUT_NAMES = 'print_output_names'

OPTION_PRINT_NOMINAL_VALUES = 'print_nominal_values'

OPTION_PRINT_BODIES = 'print_bodies'

OPTION_PRINT_HEADS = 'print_heads'

OPTION_DECIMALS_BODY = 'decimals_body'

OPTION_DECIMALS_HEAD = 'decimals_head'


class ModelWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that allow to write textual representations of models to one or several
    sinks.
    """

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write textual representations of models to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(title='Model', options=options)

    class TxtSink(OutputWriter.TxtSink):
        """
        Allows to write textual representations of models to text files.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            super().__init__(output_dir=output_dir, file_name='rules', options=options)

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)


class RuleModelWriter(ModelWriter):
    """
    Allows to write textual representations of rule-based models to one or several sinks.
    """

    class RuleModelFormattable(RuleModelVisitor, Formattable):
        """
        Allows to create textual representations of the rules in a `RuleModel`.
        """

        def __init__(self, meta_data: MetaData, model: RuleModel):
            """
            :param meta_data:   The meta-data of the training data set
            :param model:       The `RuleModel`
            """
            self.features = meta_data.features
            self.outputs = meta_data.outputs
            self.model = model
            self.text = None
            self.print_feature_names = True
            self.print_output_names = True
            self.print_nominal_values = True
            self.print_bodies = True
            self.print_heads = True
            self.body_decimals = 2
            self.head_decimals = 2

        def visit_empty_body(self, _: EmptyBody):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_empty_body`
            """
            if self.print_bodies:
                self.text.write('{}')

        def __format_conditions(self, num_conditions: int, indices: np.ndarray, thresholds: np.ndarray,
                                operator: str) -> int:
            result = num_conditions

            if indices is not None and thresholds is not None:
                text = self.text
                features = self.features
                print_feature_names = self.print_feature_names
                print_nominal_values = self.print_nominal_values
                decimals = self.body_decimals

                for i in range(indices.shape[0]):
                    if result > 0:
                        text.write(' & ')

                    feature_index = indices[i]
                    threshold = thresholds[i]
                    feature = features[feature_index] if len(features) > feature_index else None

                    if print_feature_names and feature is not None:
                        text.write(feature.name)
                    else:
                        text.write(str(feature_index))

                    text.write(' ')
                    text.write(operator)
                    text.write(' ')

                    if feature is not None and feature.nominal_values is not None:
                        nominal_value = int(threshold)

                        if print_nominal_values and len(feature.nominal_values) > nominal_value:
                            text.write('"' + feature.nominal_values[nominal_value] + '"')
                        else:
                            text.write(str(nominal_value))
                    else:
                        text.write(format_float(threshold, decimals=decimals))

                    result += 1

            return result

        def visit_conjunctive_body(self, body: ConjunctiveBody):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_conjunctive_body`
            """
            if self.print_bodies:
                text = self.text
                text.write('{')
                num_conditions = self.__format_conditions(0, body.numerical_leq_indices, body.numerical_leq_thresholds,
                                                          '<=')
                num_conditions = self.__format_conditions(num_conditions, body.numerical_gr_indices,
                                                          body.numerical_gr_thresholds, '>')
                num_conditions = self.__format_conditions(num_conditions, body.ordinal_leq_indices,
                                                          body.ordinal_leq_thresholds, '<=')
                num_conditions = self.__format_conditions(num_conditions, body.ordinal_gr_indices,
                                                          body.ordinal_gr_thresholds, '>')
                num_conditions = self.__format_conditions(num_conditions, body.nominal_eq_indices,
                                                          body.nominal_eq_thresholds, '==')
                self.__format_conditions(num_conditions, body.nominal_neq_indices, body.nominal_neq_thresholds, '!=')
                text.write('}')

        def visit_complete_head(self, head: CompleteHead):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_complete_head`
            """
            text = self.text

            if self.print_heads:
                print_output_names = self.print_output_names
                decimals = self.head_decimals
                outputs = self.outputs
                scores = head.scores

                if self.print_bodies:
                    text.write(' => ')

                text.write('(')

                for i in range(scores.shape[0]):
                    if i > 0:
                        text.write(', ')

                    if print_output_names and len(outputs) > i:
                        text.write(outputs[i].name)
                    else:
                        text.write(str(i))

                    text.write(' = ')
                    text.write(format_float(scores[i], decimals=decimals))

                text.write(')\n')
            elif self.print_bodies:
                text.write('\n')

        def visit_partial_head(self, head: PartialHead):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_partial_head`
            """
            text = self.text

            if self.print_heads:
                print_output_names = self.print_output_names
                decimals = self.head_decimals
                outputs = self.outputs
                indices = head.indices
                scores = head.scores

                if self.print_bodies:
                    text.write(' => ')

                text.write('(')

                for i in range(indices.shape[0]):
                    if i > 0:
                        text.write(', ')

                    output_index = indices[i]

                    if print_output_names and len(outputs) > output_index:
                        text.write(outputs[output_index].name)
                    else:
                        text.write(str(output_index))

                    text.write(' = ')
                    text.write(format_float(scores[i], decimals=decimals))

                text.write(')\n')
            elif self.print_bodies:
                text.write('\n')

        def format(self, options: Options, **_) -> str:
            """
            See :func:`mlrl.testbed.output_writer.Formattable.format`
            """
            self.print_feature_names = options.get_bool(OPTION_PRINT_FEATURE_NAMES, True)
            self.print_output_names = options.get_bool(OPTION_PRINT_OUTPUT_NAMES, True)
            self.print_nominal_values = options.get_bool(OPTION_PRINT_NOMINAL_VALUES, True)
            self.print_bodies = options.get_bool(OPTION_PRINT_BODIES, True)
            self.print_heads = options.get_bool(OPTION_PRINT_HEADS, True)
            self.body_decimals = options.get_int(OPTION_DECIMALS_BODY, 2)
            self.head_decimals = options.get_int(OPTION_DECIMALS_HEAD, 2)
            self.text = StringIO()
            self.model.visit_used(self)
            text = self.text.getvalue()
            self.text.close()
            return text

    # pylint: disable=unused-argument
    def _generate_output_data(self, problem_type: ProblemType, meta_data: MetaData, x, y, data_split: DataSplit,
                              learner, data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        if isinstance(learner, (ClassifierMixin, RegressorMixin)):
            model = learner.model_

            if isinstance(model, RuleModel):
                return RuleModelWriter.RuleModelFormattable(meta_data, model)

        log.error('The learner does not support to create a textual representation of the model')
        return None
