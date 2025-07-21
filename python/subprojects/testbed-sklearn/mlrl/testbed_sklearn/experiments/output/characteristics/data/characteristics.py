"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of an output matrix that are part of output data.
"""
from numbers import Number
from typing import Any, Callable, Optional, override

from mlrl.testbed_sklearn.experiments.output.characteristics.data.matrix_output import OutputMatrix

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.output.data import OutputData, OutputValue, TabularOutputData
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain
from mlrl.testbed.experiments.table import RowWiseTable, Table
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.options import Options


class Characteristic(OutputValue):
    """
    An individual characteristic that is part of output data.
    """

    Function = Callable[[Any], Number]

    def __init__(self, option_key: str, name: str, function: Function, percentage: bool = False):
        """
        :param function: The function that should be used to retrieve the value of the characteristic
        """
        super().__init__(option_key=option_key, name=name, percentage=percentage)
        self.function = function

    @override
    def format(self, value, **kwargs) -> str:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputValue.format`
        """
        return super().format(self.function(value), **kwargs)


class OutputCharacteristics(TabularOutputData):
    """
    Represents characteristics of an output matrix that are part of output data.
    """

    OPTION_OUTPUTS = 'outputs'

    OPTION_OUTPUT_DENSITY = 'output_density'

    OPTION_OUTPUT_SPARSITY = 'output_sparsity'

    OPTION_LABEL_IMBALANCE_RATIO = 'label_imbalance_ratio'

    OPTION_LABEL_CARDINALITY = 'label_cardinality'

    OPTION_DISTINCT_LABEL_VECTORS = 'distinct_label_vectors'

    def __init__(self,
                 problem_domain: ProblemDomain,
                 output_matrix: OutputMatrix,
                 properties: OutputData.Properties,
                 context: Context = Context()):
        """

        :param problem_domain:  The problem domain, the output matrix corresponds to
        :param output_matrix:   An output matrix
        :param properties:      The properties of the output data
        :param context:         A `Context` to be used by default for finding a suitable sink this output data can be
                                written to
        """
        super().__init__(properties=properties, context=context)
        self.output_matrix = output_matrix

        if isinstance(problem_domain, ClassificationProblem):
            self.characteristics = LABEL_CHARACTERISTICS
        else:
            self.characteristics = OUTPUT_CHARACTERISTICS

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        kwargs = dict(kwargs) | {OPTION_DECIMALS: 2}
        table = self.to_table(options, **kwargs)
        return table.format() if table else None

    @override
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        percentage = options.get_bool(OPTION_PERCENTAGE, kwargs.get(OPTION_PERCENTAGE, True))
        decimals = options.get_int(OPTION_DECIMALS, kwargs.get(OPTION_DECIMALS, 0))
        characteristics = Characteristic.filter_values(self.characteristics, options)
        values = map(
            lambda characteristic: characteristic.format(self.output_matrix, percentage=percentage, decimals=decimals),
            characteristics)
        return RowWiseTable(*characteristics).add_row(*values)


OUTPUT_CHARACTERISTICS = [
    Characteristic(
        option_key=OutputCharacteristics.OPTION_OUTPUTS,
        name='Outputs',
        function=lambda x: x.num_outputs,
    ),
    Characteristic(
        option_key=OutputCharacteristics.OPTION_OUTPUT_DENSITY,
        name='Output Density',
        function=lambda x: x.output_density,
        percentage=True,
    ),
    Characteristic(
        option_key=OutputCharacteristics.OPTION_OUTPUT_SPARSITY,
        name='Output Sparsity',
        function=lambda x: x.output_sparsity,
        percentage=True,
    )
]

LABEL_CHARACTERISTICS = OUTPUT_CHARACTERISTICS + [
    Characteristic(
        option_key=OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
        name='Label Imbalance Ratio',
        function=lambda x: x.avg_label_imbalance_ratio,
    ),
    Characteristic(
        option_key=OutputCharacteristics.OPTION_LABEL_CARDINALITY,
        name='Label Cardinality',
        function=lambda x: x.avg_label_cardinality,
    ),
    Characteristic(
        option_key=OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
        name='Distinct Label Vectors',
        function=lambda x: x.num_distinct_label_vectors,
    )
]
