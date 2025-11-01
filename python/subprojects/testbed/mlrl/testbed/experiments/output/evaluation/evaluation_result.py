"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing evaluation results that are part of output data.
"""
from functools import partial
from itertools import chain
from typing import Dict, Iterable, List, Optional, Set, Tuple, override

import numpy as np

from scipy.stats import rankdata

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import TabularProperties
from mlrl.testbed.experiments.output.data import OutputValue, TabularOutputData
from mlrl.testbed.experiments.output.evaluation.measures import AggregationMeasure, Measure
from mlrl.testbed.experiments.table import Cell, Column, ColumnWiseTable, RowWiseTable, Table
from mlrl.testbed.util.format import OPTION_DECIMALS, format_number

from mlrl.util.options import Options


class AggregatedEvaluationResult(TabularOutputData):
    """
    Stores evaluation results that have been aggregated across several experiments.
    """

    PROPERTIES = TabularProperties(name='Evaluation results of several experiments', file_name='aggregated_evaluation')

    CONTEXT = Context(include_prediction_scope=False, include_fold=False)

    OPTION_ENABLE_ALL = 'enable_all'

    OPTION_RANK = 'rank'

    COLUMN_DATASET = 'Dataset'

    COLUMN_PREFIX_PARAMETER = 'Parameter'

    AGGREGATION_MEASURES = [
        AggregationMeasure(
            option_key=OPTION_RANK,
            name='Rank',
            aggregation_function=lambda values, smaller_is_better: rankdata(np.asarray(values) *
                                                                            (1 if smaller_is_better else -1),
                                                                            method='average'),
        ),
    ]

    def __init__(self, evaluation_by_dataset: Dict[str, Table]):
        """
        :param evaluation_by_dataset: A dictionary that stores a table with evaluation results, mapped to the names of
                                      different datasets
        """
        super().__init__(properties=self.PROPERTIES, context=self.CONTEXT)
        self.evaluation_by_dataset = evaluation_by_dataset

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        text = ''
        kwargs = dict(kwargs) | {OPTION_DECIMALS: 2}
        table = self.to_table(options, **kwargs)

        if table:
            column_wise_table = table.to_column_wise_table()
            dataset_column_index = 0
            parameter_column_indices: List[int] = []
            measures: List[Tuple[int, str]] = []
            std_dev_column_indices: Dict[str, int] = {}
            aggregation_measure_column_indices: Dict[str, Dict[AggregationMeasure, int]] = {}

            for column_index, column in enumerate(column_wise_table.columns):
                header = str(column.header)
                aggregation_measure = next(
                    (measure for measure in self.AGGREGATION_MEASURES if header.startswith(measure.name)), None)

                if aggregation_measure:
                    key = header[len(aggregation_measure.name):].lstrip()
                    dictionary = aggregation_measure_column_indices.setdefault(key, {})
                    dictionary[aggregation_measure] = column_index
                    column.set_header(aggregation_measure.name)
                elif header == self.COLUMN_DATASET:
                    dataset_column_index = column_index
                elif header.startswith(self.COLUMN_PREFIX_PARAMETER):
                    parameter_column_indices.append(column_index)
                    column.set_header(header[len(self.COLUMN_PREFIX_PARAMETER):].lstrip().lstrip('-').replace('-', '_'))
                elif header.startswith(OutputValue.COLUMN_PREFIX_STD_DEV):
                    key = header[len(OutputValue.COLUMN_PREFIX_STD_DEV):].lstrip()
                    std_dev_column_indices[key] = column_index
                    column.set_header(OutputValue.COLUMN_PREFIX_STD_DEV)
                else:
                    measures.append((column_index, header))

            measures.sort(key=lambda x: x[1])

            for i, (measure_index, measure) in enumerate(measures):
                if i > 0:
                    text += '\n\n'

                text += 'Evaluation results for measure "' + measure + '":\n\n'
                std_dev_index = std_dev_column_indices.get(measure)
                aggregation_measure_indices = aggregation_measure_column_indices.get(measure, {})
                measure_indices = [measure_index] + ([std_dev_index] if std_dev_index else [])
                relevant_indices = chain(parameter_column_indices, measure_indices)
                sliced_table = column_wise_table.slice(*relevant_indices)

                if std_dev_index:
                    self.__add_std_dev_column(sliced_table)

                for aggregation_measure, column_index, in aggregation_measure_indices.items():
                    sliced_table.add_column(*column_wise_table[column_index], header=aggregation_measure.name)

                row_wise_table = sliced_table.to_row_wise_table()
                dataset_column = column_wise_table[dataset_column_index]
                parameter_columns: List[Column] = [
                    column_wise_table[column_index] for column_index in parameter_column_indices
                ]
                average_rows = self.__get_average_rows(table=column_wise_table,
                                                       aggregation_measure_indices=aggregation_measure_indices,
                                                       parameter_columns=parameter_columns,
                                                       num_columns=row_wise_table.num_columns,
                                                       decimals=options.get_int(OPTION_DECIMALS,
                                                                                kwargs.get(OPTION_DECIMALS, 0)))
                separators = self.__add_separator_rows(dataset_column, row_wise_table, averages=bool(average_rows))

                for row in average_rows:
                    row_wise_table.add_row(*row)

                separator_indices = [
                    row_index for row_index, row in enumerate(row_wise_table.rows)
                    if row_index > 0 and (row[0] in separators or row_wise_table[row_index - 1][0] in separators)
                ]
                text += row_wise_table.format(table_format=Table.Format.SIMPLE, separator_indices=separator_indices)

        return text if text else None

    # pylint: disable=too-many-nested-blocks
    @staticmethod
    def __get_average_rows(table: ColumnWiseTable, aggregation_measure_indices: Dict[AggregationMeasure, int],
                           parameter_columns: List[Column], num_columns: int, decimals: int) -> List[List[Cell]]:
        result = []

        for parameter_setting, row_indices in AggregatedEvaluationResult.__get_unique_parameter_settings(
                parameter_columns).items():
            row = list(parameter_setting) + ([None] *
                                             (num_columns - len(parameter_setting) - len(aggregation_measure_indices)))

            for aggregation_measure, column_index, in aggregation_measure_indices.items():
                values: List[float] = []

                if aggregation_measure.can_be_averaged:
                    for row_index in row_indices:
                        value = table[column_index][row_index]

                        if value is not None:
                            try:
                                values.append(float(value))
                            except ValueError:
                                pass

                row.append(format_number(np.asarray(values, dtype=float).mean(), decimals=decimals) if values else None)

            result.append(row)

        return result

    @staticmethod
    def __get_unique_parameter_settings(parameter_columns: List[Column]) -> Dict[Tuple[Cell, ...], List[int]]:
        num_rows = parameter_columns[0].num_rows if parameter_columns else 0
        unique_parameters: Dict[Tuple[Cell, ...], List[int]] = {}

        for row_index in range(num_rows):
            parameters = tuple((parameter_column[row_index] for parameter_column in parameter_columns))
            unique_parameters.setdefault(parameters, []).append(row_index)

        return unique_parameters

    @staticmethod
    def __add_std_dev_column(sliced_table: ColumnWiseTable):
        std_dev_column = sliced_table[sliced_table.num_columns - 1]

        for row_index in range(std_dev_column.num_rows):
            std_dev_column[row_index] = '±' + str(std_dev_column[row_index])

    @staticmethod
    def __add_separator_rows(dataset_column: Column, table: RowWiseTable, averages: bool = False) -> Set[str]:
        previous_dataset: Optional[str] = None
        separators: Set[str] = set()

        for row_index in range(dataset_column.num_rows - 1, -1, -1):
            dataset = '"' + str(dataset_column[row_index]) + '"'

            if row_index == 0:
                table.add_row(dataset, position=0)

            if previous_dataset and dataset != previous_dataset:
                table.add_row(previous_dataset, position=row_index + 1)

            previous_dataset = dataset
            separators.add(dataset)

        if averages:
            separator_averages = 'Averages'
            table.add_row(separator_averages)
            separators.add(separator_averages)

        return separators

    @override
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        evaluation_by_dataset = self.evaluation_by_dataset

        if evaluation_by_dataset:
            dataset_names = sorted(evaluation_by_dataset.keys())
            tables: List[Table] = []
            dataset_column: List[str] = []

            for dataset_name in dataset_names:
                table = evaluation_by_dataset[dataset_name]
                tables.append(table)
                dataset_column.extend((dataset_name for _ in range(table.num_rows)))

            aggregated_table = RowWiseTable.aggregate(*tables).to_column_wise_table()
            aggregated_table.add_column(*dataset_column, header=self.COLUMN_DATASET, position=0)
            decimals = options.get_int(OPTION_DECIMALS, kwargs.get(OPTION_DECIMALS, 0))
            aggregation_measures = OutputValue.filter_values(self.AGGREGATION_MEASURES, options)

            for column_index in range(aggregated_table.num_columns - 1, -1, -1):
                column = aggregated_table[column_index]
                header = str(column.header)
                current_dataset: Optional[str] = None
                values_by_dataset: List[List[float]] = []

                for row_index in range(column.num_rows):
                    dataset = dataset_column[row_index] if row_index < len(dataset_column) else None

                    if dataset and dataset != current_dataset:
                        current_dataset = dataset
                        values_by_dataset.append([])

                    try:
                        value = column[row_index]

                        if value is not None:
                            float_value = float(value)
                            values_by_dataset[-1].append(float_value)
                            column[row_index] = format_number(float_value, decimals=decimals)
                    except ValueError:
                        pass

                if header != self.COLUMN_DATASET \
                        and not header.startswith(self.COLUMN_PREFIX_PARAMETER) \
                        and not header.startswith(OutputValue.COLUMN_PREFIX_STD_DEV) \
                        and values_by_dataset:
                    smaller_is_better = Measure.is_smaller_better(header)

                    for aggregation_measure in aggregation_measures:
                        if isinstance(aggregation_measure, AggregationMeasure):

                            def aggregation_function(aggregation_measure: AggregationMeasure, smaller_is_better: bool,
                                                     values_list: List[float]) -> Iterable[float]:
                                return aggregation_measure.aggregate(values_list, smaller_is_better=smaller_is_better)

                            aggregated_column = chain.from_iterable(
                                map(partial(aggregation_function, aggregation_measure, smaller_is_better),
                                    values_by_dataset))
                            aggregated_table.add_column(*map(lambda x: format_number(x, decimals=decimals),
                                                             aggregated_column),
                                                        header=f'{aggregation_measure.name} {header}',
                                                        position=column_index + 1)

            return aggregated_table

        return None
