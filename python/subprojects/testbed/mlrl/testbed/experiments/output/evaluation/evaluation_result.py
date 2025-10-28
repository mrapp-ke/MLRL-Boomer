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
from mlrl.testbed.experiments.table import Column, ColumnWiseTable, RowWiseTable, Table
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
            aggregation_measure_column_indices: Dict[str, Dict[str, int]] = {}

            for column_index, column in enumerate(column_wise_table.columns):
                header = str(column.header)
                aggregation_measure_names = map(lambda x: x.name, self.AGGREGATION_MEASURES)
                aggregation_measure = next((name for name in aggregation_measure_names if header.startswith(name)),
                                           None)

                if aggregation_measure:
                    key = header[len(aggregation_measure):].lstrip()
                    dictionary = aggregation_measure_column_indices.setdefault(key, {})
                    dictionary[aggregation_measure] = column_index
                    column.set_header(aggregation_measure)
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
                relevant_indices = chain(parameter_column_indices, [measure_index] +
                                         ([std_dev_index] if std_dev_index else []))
                sliced_table = column_wise_table.slice(*relevant_indices)

                if std_dev_index:
                    self.__add_std_dev_column(sliced_table)

                for aggregation_measure, column_index, in aggregation_measure_indices.items():
                    sliced_table.add_column(*column_wise_table[column_index], header=aggregation_measure)

                row_wise_table = sliced_table.to_row_wise_table()
                dataset_column = column_wise_table[dataset_column_index]
                datasets = self.__add_dataset_rows(dataset_column, row_wise_table)
                separator_indices = [
                    row_index for row_index, row in enumerate(row_wise_table.rows)
                    if row_index > 0 and (row[0] in datasets or row_wise_table[row_index - 1][0] in datasets)
                ]
                text += row_wise_table.format(table_format=Table.Format.SIMPLE, separator_indices=separator_indices)

        return text if text else None

    @staticmethod
    def __add_std_dev_column(sliced_table: ColumnWiseTable):
        std_dev_column = sliced_table[sliced_table.num_columns - 1]

        for row_index in range(std_dev_column.num_rows):
            std_dev_column[row_index] = '±' + str(std_dev_column[row_index])

    @staticmethod
    def __add_dataset_rows(dataset_column: Column, table: RowWiseTable) -> set[str]:
        previous_dataset: Optional[str] = None
        datasets: Set[str] = set()

        for row_index in range(dataset_column.num_rows - 1, -1, -1):
            dataset = '"' + str(dataset_column[row_index]) + '"'

            if row_index == 0:
                table.add_row(dataset, position=0)

            if previous_dataset and dataset != previous_dataset:
                table.add_row(previous_dataset, position=row_index + 1)

            previous_dataset = dataset
            datasets.add(dataset)

        return datasets

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
