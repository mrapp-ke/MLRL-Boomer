"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing rule models as text that is part of output data.
"""
from io import StringIO
from typing import Optional

import numpy as np

from mlrl.common.cython.rule_model import CompleteHead, ConjunctiveBody, EmptyBody, PartialHead, RuleModel, \
    RuleModelVisitor

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.output.data import OutputData, TextualOutputData
from mlrl.testbed.util.format import format_number

from mlrl.util.options import Options


class RuleModelAsText(TextualOutputData):
    """
    A textual representation of a rule model.
    """

    OPTION_PRINT_FEATURE_NAMES = 'print_feature_names'

    OPTION_PRINT_OUTPUT_NAMES = 'print_output_names'

    OPTION_PRINT_NOMINAL_VALUES = 'print_nominal_values'

    OPTION_PRINT_BODIES = 'print_bodies'

    OPTION_PRINT_HEADS = 'print_heads'

    OPTION_DECIMALS_BODY = 'decimals_body'

    OPTION_DECIMALS_HEAD = 'decimals_head'

    class Visitor(RuleModelVisitor):
        """
        Access the individual rules in a `RuleModel` and creates textual representations of these rules.
        """

        def __format_conditions(self, indices: np.ndarray, thresholds: np.ndarray, operator: str, offset: int) -> int:
            if indices is not None and thresholds is not None:
                text = self.text
                features = self.dataset.features
                print_feature_names = self.print_feature_names
                print_nominal_values = self.print_nominal_values
                decimals = self.body_decimals

                for i in range(indices.shape[0]):
                    if offset > 0:
                        text.write(' & ')

                    feature_index = indices[i]
                    threshold = thresholds[i]
                    feature = features[feature_index] if len(features) > feature_index else None

                    if print_feature_names and feature:
                        text.write(feature.name)
                    else:
                        text.write(str(feature_index))

                    text.write(' ')
                    text.write(operator)
                    text.write(' ')

                    if feature and feature.nominal_values:
                        nominal_value = int(threshold)

                        if print_nominal_values and len(feature.nominal_values) > nominal_value:
                            text.write('"' + feature.nominal_values[nominal_value] + '"')
                        else:
                            text.write(str(nominal_value))
                    else:
                        text.write(format_number(threshold, decimals=decimals))

                    offset += 1

            return offset

        def __format_numerical_conditions(self, body: ConjunctiveBody, offset: int) -> int:
            offset = self.__format_conditions(offset=offset,
                                              indices=body.numerical_leq_indices,
                                              thresholds=body.numerical_leq_thresholds,
                                              operator='<=')
            offset = self.__format_conditions(offset=offset,
                                              indices=body.numerical_gr_indices,
                                              thresholds=body.numerical_gr_thresholds,
                                              operator='>')
            return offset

        def __format_ordinal_conditions(
            self,
            body: ConjunctiveBody,
            offset: int,
        ) -> int:
            offset = self.__format_conditions(offset=offset,
                                              indices=body.ordinal_leq_indices,
                                              thresholds=body.ordinal_leq_thresholds,
                                              operator='<=')
            offset = self.__format_conditions(offset=offset,
                                              indices=body.ordinal_gr_indices,
                                              thresholds=body.ordinal_gr_thresholds,
                                              operator='>')
            return offset

        def __format_nominal_conditions(self, body: ConjunctiveBody, offset: int) -> int:
            offset = self.__format_conditions(offset=offset,
                                              indices=body.nominal_eq_indices,
                                              thresholds=body.nominal_eq_thresholds,
                                              operator='==')
            offset = self.__format_conditions(offset=offset,
                                              indices=body.nominal_neq_indices,
                                              thresholds=body.nominal_neq_thresholds,
                                              operator='!=')
            return offset

        def __init__(self, dataset: TabularDataset, options: Options):
            """
            :param dataset: The dataset on which the model has been trained
            """
            self.dataset = dataset
            self.print_feature_names = options.get_bool(RuleModelAsText.OPTION_PRINT_FEATURE_NAMES, True)
            self.print_output_names = options.get_bool(RuleModelAsText.OPTION_PRINT_OUTPUT_NAMES, True)
            self.print_nominal_values = options.get_bool(RuleModelAsText.OPTION_PRINT_NOMINAL_VALUES, True)
            self.print_bodies = options.get_bool(RuleModelAsText.OPTION_PRINT_BODIES, True)
            self.print_heads = options.get_bool(RuleModelAsText.OPTION_PRINT_HEADS, True)
            self.body_decimals = options.get_int(RuleModelAsText.OPTION_DECIMALS_BODY, 2)
            self.head_decimals = options.get_int(RuleModelAsText.OPTION_DECIMALS_HEAD, 2)
            self.text = StringIO()

        def visit_empty_body(self, _: EmptyBody):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_empty_body`
            """
            if self.print_bodies:
                self.text.write('{}')

        def visit_conjunctive_body(self, body: ConjunctiveBody):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_conjunctive_body`
            """
            if self.print_bodies:
                text = self.text
                text.write('{')
                offset = self.__format_numerical_conditions(body, offset=0)
                offset = self.__format_ordinal_conditions(body, offset=offset)
                self.__format_nominal_conditions(body, offset=offset)
                text.write('}')

        def visit_complete_head(self, head: CompleteHead):
            """
            See :func:`mlrl.common.cython.rule_model.RuleModelVisitor.visit_complete_head`
            """
            text = self.text

            if self.print_heads:
                print_output_names = self.print_output_names
                decimals = self.head_decimals
                outputs = self.dataset.outputs
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
                    text.write(format_number(scores[i], decimals=decimals))

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
                outputs = self.dataset.outputs
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
                    text.write(format_number(scores[i], decimals=decimals))

                text.write(')\n')
            elif self.print_bodies:
                text.write('\n')

    def __init__(self, model: RuleModel, dataset: TabularDataset):
        """
        :param model:   The rule model
        :param dataset: The dataset on which the model has been trained
        """
        super().__init__(OutputData.Properties(name='Model', file_name='rules'), Context(include_dataset_type=False))
        self.model = model
        self.dataset = dataset

    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        visitor = RuleModelAsText.Visitor(self.dataset, options)
        self.model.visit_used(visitor)
        return visitor.text.getvalue()
