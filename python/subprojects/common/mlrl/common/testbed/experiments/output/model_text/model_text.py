"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing rule models as text that is part of output data.
"""
from io import StringIO
from typing import Optional, override

from mlrl.common.cython.rule_model import Body, CompleteHead, ConjunctiveBody, EmptyBody, Head, PartialHead, RuleModel

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

    @staticmethod
    def __format_rule_model(rule_model: RuleModel, dataset: TabularDataset, options: Options) -> str:
        text = StringIO()

        for rule in rule_model:
            RuleModelAsText.__format_body(text, rule.body, dataset, options)
            RuleModelAsText.__format_head(text, rule.head, dataset, options)

        return text.getvalue()

    @staticmethod
    def __format_body(text: StringIO, body: Body, dataset: TabularDataset, options: Options):
        if options.get_bool(RuleModelAsText.OPTION_PRINT_BODIES, True):
            if isinstance(body, EmptyBody):
                text.write('{}')
            elif isinstance(body, ConjunctiveBody):
                text.write('{')
                RuleModelAsText.__format_conditions(text, body, dataset, options)
                text.write('}')
            else:
                raise ValueError('Unsupported type of body: ' + str(type(body)))

    @staticmethod
    def __format_head(text: StringIO, head: Head, dataset: TabularDataset, options: Options):
        if isinstance(head, CompleteHead):
            RuleModelAsText.__format_complete_head(text, head, dataset, options)
        elif isinstance(head, PartialHead):
            RuleModelAsText.__format_partial_head(text, head, dataset, options)
        else:
            raise ValueError('Unsupported type of head: ' + str(type(head)))

    @staticmethod
    def __format_conditions(text: StringIO, body: Body, dataset: TabularDataset, options: Options):
        print_feature_names = options.get_bool(RuleModelAsText.OPTION_PRINT_FEATURE_NAMES, True)
        print_nominal_values = options.get_bool(RuleModelAsText.OPTION_PRINT_NOMINAL_VALUES, True)
        decimals = options.get_int(RuleModelAsText.OPTION_DECIMALS_BODY, 2)
        features = dataset.features

        for i, condition in enumerate(body):
            if i > 0:
                text.write(' & ')

            feature_index = condition.feature_index
            threshold = condition.threshold
            feature = features[feature_index] if len(features) > feature_index else None

            if print_feature_names and feature:
                text.write(feature.name)
            else:
                text.write(str(feature_index))

            text.write(' ')
            text.write(str(condition.comparator))
            text.write(' ')

            if feature and feature.nominal_values:
                nominal_value = int(threshold)

                if print_nominal_values and len(feature.nominal_values) > nominal_value:
                    text.write('"' + feature.nominal_values[nominal_value] + '"')
                else:
                    text.write(str(nominal_value))
            else:
                text.write(format_number(threshold, decimals=decimals))

    @staticmethod
    def __format_complete_head(text: StringIO, head: CompleteHead, dataset: TabularDataset, options: Options):
        print_bodies = options.get_bool(RuleModelAsText.OPTION_PRINT_BODIES, True)

        if options.get_bool(RuleModelAsText.OPTION_PRINT_HEADS, True):
            print_output_names = options.get_bool(RuleModelAsText.OPTION_PRINT_OUTPUT_NAMES, True)
            decimals = options.get_int(RuleModelAsText.OPTION_DECIMALS_HEAD, 2)
            outputs = dataset.outputs

            if print_bodies:
                text.write(' => ')

            text.write('(')

            for i, prediction in enumerate(head):
                if i > 0:
                    text.write(', ')

                output_index = prediction.output_index
                text.write(outputs[output_index].name if print_output_names and len(outputs) > i else str(output_index))
                text.write(' = ')
                text.write(format_number(prediction.value, decimals=decimals))

            text.write(')\n')
        elif print_bodies:
            text.write('\n')

    @staticmethod
    def __format_partial_head(text: StringIO, head: PartialHead, dataset: TabularDataset, options: Options):
        print_bodies = options.get_bool(RuleModelAsText.OPTION_PRINT_BODIES, True)

        if options.get_bool(RuleModelAsText.OPTION_PRINT_HEADS, True):
            print_output_names = options.get_bool(RuleModelAsText.OPTION_PRINT_OUTPUT_NAMES, True)
            decimals = options.get_int(RuleModelAsText.OPTION_DECIMALS_HEAD, 2)
            outputs = dataset.outputs

            if print_bodies:
                text.write(' => ')

            text.write('(')

            for i, prediction in enumerate(head):
                if i > 0:
                    text.write(', ')

                output_index = prediction.output_index

                if print_output_names and len(outputs) > output_index:
                    text.write(outputs[output_index].name)
                else:
                    text.write(str(output_index))

                text.write(' = ')
                text.write(format_number(prediction.value, decimals=decimals))

            text.write(')\n')
        elif print_bodies:
            text.write('\n')

    def __init__(self, model: RuleModel, dataset: TabularDataset):
        """
        :param model:   The rule model
        :param dataset: The dataset on which the model has been trained
        """
        super().__init__(OutputData.Properties(name='Model', file_name='rules'), Context(include_dataset_type=False))
        self.model = model
        self.dataset = dataset

    @override
    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        return self.__format_rule_model(self.model, self.dataset, options)
