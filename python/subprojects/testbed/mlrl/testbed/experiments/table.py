"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing tables.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generator, Iterable, Iterator, List, Optional, override

from tabulate import tabulate


class Alignment(Enum):
    """
    All possible alignments of a table column.
    """
    AUTO = None
    LEFT = 'left'
    CENTER = 'center'
    RIGHT = 'right'


Header = Optional[Any]

Cell = Optional[Any]


class HeaderRow(Iterable[Optional[Any]], ABC):
    """
    Provides access to the headers of a table.
    """

    @property
    @abstractmethod
    def num_columns(self) -> int:
        """
        The number of columns in the table.
        """

    @abstractmethod
    def __getitem__(self, column_index: int) -> Header:
        """
        Returns the header of the column at a specific index.

        :param column_index:    The index of the column
        :return:                The header at the given index
        """


class Row(Iterable[Cell], ABC):
    """
    Provides access to a single row of a table.
    """

    @property
    @abstractmethod
    def num_columns(self) -> int:
        """
        The number of columns in the table.
        """

    @abstractmethod
    def __getitem__(self, column_index: int) -> Cell:
        """
        Returns the cell of the column at a specific index.

        :param column_index:    The index of the column
        :return:                The cell at the given index
        """


class Column(Iterable[Cell], ABC):
    """
    Provides access to a single column of a table.
    """

    @property
    @abstractmethod
    def header(self) -> Optional[Header]:
        """
        The header of the column.
        """

    @property
    @abstractmethod
    def num_rows(self) -> int:
        """
        The number of rows in the table.
        """

    @abstractmethod
    def __getitem__(self, row_index: int) -> Cell:
        """
        Returns the cell of the row at a specific index.

        :param row_index:   The index of the row
        :return:            The cell at the given index
        """


class Table(ABC):
    """
    An abstract base class for all tables.
    """

    @property
    def has_headers(self) -> bool:
        """
        True, if the table has any headers, False otherwise.
        """
        return self.header_row is not None

    @property
    @abstractmethod
    def num_rows(self) -> int:
        """
        The number of rows in the table.
        """

    @property
    @abstractmethod
    def num_columns(self) -> int:
        """
        The number of columns in the table.
        """

    @property
    def num_cells(self) -> int:
        """
        The number of cells in the table.
        """
        return self.num_rows * self.num_columns

    @property
    @abstractmethod
    def header_row(self) -> Optional[HeaderRow]:
        """
        The header row of the table or None, if the table does not have any headers.
        """

    @property
    @abstractmethod
    def alignments(self) -> Optional[List[Alignment]]:
        """
        The individual alignments of the columns in the table or None, if no alignments have been specified.
        """

    @property
    @abstractmethod
    def rows(self) -> Generator[Row, None, None]:
        """
        A generator that provides access to the individual rows in the table.
        """

    @property
    @abstractmethod
    def columns(self) -> Generator[Column, None, None]:
        """
        A generator that provides access to the individual columns in the table.
        """

    @abstractmethod
    def to_row_wise_table(self) -> 'RowWiseTable':
        """
        Creates and returns a `RowWiseTable` from this table.

        :return: The `RowWiseTable` that has been created
        """

    @abstractmethod
    def to_column_wise_table(self) -> 'ColumnWiseTable':
        """
        Creates and returns a `ColumnWiseTable` from this table.

        :return: The `ColumnWiseTable` that has been created
        """

    def format(self, auto_rotate: bool = True) -> str:
        """
        Creates and returns a textual representation of the table.

        :param auto_rotate: True, if tables with a single row should automatically be rotated for legibility, False
                            otherwise
        :return:            The textual representation that has been created
        """
        if self.num_cells > 0:
            headers = self.header_row
            rows = self.rows

            if auto_rotate and headers and self.num_rows == 1:
                first_row = next(self.rows)
                rotated_rows = [[headers[column_index], first_row[column_index]]
                                for column_index in range(self.num_columns)]
                return tabulate(rotated_rows, tablefmt='plain')

            if not headers:
                return tabulate(rows, tablefmt='plain')

            alignments = map(lambda alignment: alignment.value, self.alignments) if self.alignments else None
            return tabulate(rows, headers=headers, tablefmt='simple_outline', colalign=alignments)

        return ''


class RowWiseTable(Table):
    """
    A table to which new rows can be added one by one.
    """

    class HeaderRow(HeaderRow):
        """
        Provides access to the headers of a `RowWiseTable`.
        """

        def __init__(self, table: 'RowWiseTable'):
            """
            :param table: The `RowWiseTable` to be accessed
            """
            self.table = table

        @override
        @property
        def num_columns(self) -> int:
            return self.table.num_columns

        @override
        def __getitem__(self, column_index: int) -> Header:
            return self.table._headers[column_index]

        @override
        def __iter__(self) -> Iterator[Optional[Any]]:
            return iter(self.table._headers)

    class Row(Row):
        """
        Provides access to a single row of a `RowWiseTable`.
        """

        def __init__(self, table: 'RowWiseTable', row_index: int):
            """
            :param table:       The `RowWiseTable` to be accessed
            :param row_index:   The index of the row to be accessed
            """
            self.table = table
            self.row_index = row_index

        @override
        @property
        def num_columns(self) -> int:
            return self.table.num_columns

        @override
        def __getitem__(self, column_index: int) -> Cell:
            if column_index >= self.num_columns:
                raise IndexError('Invalid column index: Got ' + str(column_index) + ', but table has only '
                                 + str(self.num_columns) + ' columns')

            row = self.table._rows[self.row_index]
            return row[column_index] if column_index < len(row) else None

        @override
        def __iter__(self) -> Iterator[Cell]:
            for column_index in range(self.num_columns):
                yield self[column_index]

    class Column(Column):
        """
        Provides access to a single column of a `RowWiseTable`.
        """

        def __init__(self, table: 'RowWiseTable', column_index: int):
            """
            :param table:           The `RowWiseTable` to be accessed
            :param column_index:    The index of the column to be accessed
            """
            self.table = table
            self.column_index = column_index

        @override
        @property
        def header(self) -> Optional[Header]:
            header_row = self.table.header_row
            return header_row[self.column_index] if header_row else None

        @override
        @property
        def num_rows(self) -> int:
            return self.table.num_rows

        @override
        def __getitem__(self, row_index: int) -> Cell:
            if row_index >= self.num_rows:
                raise IndexError('Invalid row index: Got ' + str(row_index) + ', but table has only '
                                 + str(self.num_rows) + ' rows')

            row = self.table._rows[row_index]
            column_index = self.column_index
            return row[column_index] if column_index < len(row) else None

        @override
        def __iter__(self) -> Iterator[Cell]:
            for row_index in range(self.num_rows):
                yield self[row_index]

    def __init__(self, *headers: Header, alignments: Optional[List[Alignment]] = None):
        """
        :param headers:     The headers of the individual columns
        :param alignments:  The alignments of the individual columns or None
        """
        if headers and alignments and len(headers) != len(alignments):
            raise ValueError('The number of alignments does not match the number headers: Expected ' + str(len(headers))
                             + ', but got ' + str(len(alignments)))

        self._headers = headers
        self._alignments = alignments
        self._rows: List[List[Any]] = []
        self._num_columns = len(headers) if headers else 0

    def add_row(self, *values: Optional[Any]):
        """
        Adds a new row to the end of the table.

        :param values:  The values in the row
        :return:        The modified table
        """
        row = list(values)
        headers = self._headers

        if headers and len(row) > len(headers):
            raise ValueError('The row must contain at most ' + str(len(headers)) + ' values')

        self._rows.append(row)
        self._num_columns = max(len(row), self._num_columns)
        return self

    def sort_by_columns(self,
                        column_index: int,
                        *additional_column_indices: int,
                        descending: bool = False) -> 'RowWiseTable':
        """
        Sorts the rows in the table by the values in one or several columns.

        :param column_index:                The index of the column to sort by
        :param additional_column_indices:   Additional indices of columns to sort by
        :param descending:                  True, if the rows should be sorted in descending order, False, if they
                                            should be sorted in ascending order
        :return:                            The sorted table
        """
        self._rows.sort(key=lambda row: ([row[i] for i in [column_index] + list(additional_column_indices)]),
                        reverse=descending)
        return self

    @override
    @property
    def num_rows(self) -> int:
        return len(self._rows)

    @override
    @property
    def num_columns(self) -> int:
        return self._num_columns

    @override
    @property
    def header_row(self) -> Optional[HeaderRow]:
        return RowWiseTable.HeaderRow(self) if self._headers else None

    @override
    @property
    def alignments(self) -> Optional[List[Alignment]]:
        return self._alignments

    @override
    @property
    def rows(self) -> Generator[Row, None, None]:
        for row_index in range(self.num_rows):
            yield RowWiseTable.Row(self, row_index)

    @override
    @property
    def columns(self) -> Generator[Column, None, None]:
        for column_index in range(self.num_columns):
            yield RowWiseTable.Column(self, column_index)

    @override
    def to_row_wise_table(self) -> 'RowWiseTable':
        return self

    @override
    def to_column_wise_table(self) -> 'ColumnWiseTable':
        table = ColumnWiseTable()
        header_row = self.header_row
        alignments = self.alignments

        for column_index, column in enumerate(self.columns):
            table.add_column(*column,
                             header=header_row[column_index] if header_row else None,
                             alignment=alignments[column_index] if alignments else None)

        return table


class ColumnWiseTable(Table):
    """
    A table to which new columns can be added one by one.
    """

    class HeaderRow(HeaderRow):
        """
        Provides access to the headers of a `ColumnWiseTable`.
        """

        def __init__(self, table: 'ColumnWiseTable'):
            """
            :param table: The `ColumnWiseTable` to be accessed
            """
            self.table = table

        @override
        @property
        def num_columns(self) -> int:
            return self.table.num_columns

        @override
        def __getitem__(self, column_index: int) -> Header:
            return self.table._headers[column_index]

        @override
        def __iter__(self) -> Iterator[Header]:
            return iter(self.table._headers)

    class Row(Row):
        """
        Provides access to a single row of a `ColumnWiseTable`.
        """

        def __init__(self, table: 'ColumnWiseTable', row_index: int):
            """
            :param table:       The `ColumnWiseTable` to be accessed
            :param row_index:   The index of the row to be accessed
            """
            self.table = table
            self.row_index = row_index

        @override
        @property
        def num_columns(self) -> int:
            return self.table.num_columns

        @override
        def __getitem__(self, column_index: int) -> Cell:
            if column_index >= self.num_columns:
                raise IndexError('Invalid column index: Got ' + str(column_index) + ', but table has only '
                                 + str(self.num_columns) + ' columns')

            column = self.table._columns[column_index]
            row_index = self.row_index
            return column[row_index] if row_index < len(column) else None

        @override
        def __iter__(self) -> Iterator[Cell]:
            for column_index in range(self.num_columns):
                yield self[column_index]

    class Column(Column):
        """
        Provides access to a single column of a `ColumnWiseTable`.
        """

        def __init__(self, table: 'ColumnWiseTable', column_index: int):
            """
            :param table:           The `ColumnWiseTable` to be accessed
            :param column_index:    The index of the column to be accessed
            """
            self.table = table
            self.column_index = column_index

        @override
        @property
        def header(self) -> Optional[Header]:
            header_row = self.table.header_row
            return header_row[self.column_index] if header_row else None

        @override
        @property
        def num_rows(self) -> int:
            return self.table.num_rows

        @override
        def __getitem__(self, row_index: int) -> Cell:
            if row_index >= self.num_rows:
                raise IndexError('Invalid row index: Got ' + str(row_index) + ', but table has only '
                                 + str(self.num_rows) + ' rows')

            column = self.table._columns[self.column_index]
            return column[row_index] if row_index < len(column) else None

        @override
        def __iter__(self) -> Iterator[Cell]:
            for row_index in range(self.num_rows):
                yield self[row_index]

    def __init__(self):
        self._headers = None
        self._alignments = None
        self._columns = []
        self._num_rows = 0

    def add_column(self,
                   *values: Optional[Any],
                   header: Optional[Header] = None,
                   alignment: Optional[Alignment] = None) -> 'ColumnWiseTable':
        """
        Adds a new column to the end of the table.

        :param values:      The values in the column
        :param header:      The header of the column or None
        :param alignment:   The alignment of the column or None
        :return:            The modified table
        """
        column = list(values)

        if header:
            if not self._headers:
                self._headers = [None for _ in range(self.num_columns)]

            self._headers.append(header)

        if alignment:
            if not self._alignments:
                self._alignments = [None for _ in range(self.num_columns)]

            self._alignments.append(alignment)

        self._columns.append(column)
        self._num_rows = max(len(column), self._num_rows)
        return self

    def sort_by_headers(self) -> 'ColumnWiseTable':
        """
        Sorts the columns in the table by their headers.

        :return: The sorted table
        """
        headers = self._headers

        if headers:
            columns = self._columns
            alignments = self._alignments
            sorted_column_indices = sorted(range(len(headers)), key=headers.__getitem__)
            sorted_columns = []
            sorted_headers = []
            sorted_alignments: Optional[List[str]] = [] if alignments else None

            for column_index in sorted_column_indices:
                sorted_columns.append(columns[column_index])
                sorted_headers.append(headers[column_index])

                if sorted_alignments and alignments:
                    sorted_alignments.append(alignments[column_index])

            self._headers = sorted_headers
            self._alignments = sorted_alignments
            self._columns = sorted_columns

        return self

    @override
    @property
    def num_rows(self) -> int:
        return self._num_rows

    @override
    @property
    def num_columns(self) -> int:
        return len(self._columns)

    @override
    @property
    def header_row(self) -> Optional[HeaderRow]:
        return ColumnWiseTable.HeaderRow(self) if self._headers else None

    @override
    @property
    def alignments(self) -> Optional[List[Alignment]]:
        return self._alignments

    @override
    @property
    def rows(self) -> Generator[Row, None, None]:
        for row_index in range(self.num_rows):
            yield ColumnWiseTable.Row(self, row_index)

    @override
    @property
    def columns(self) -> Generator[Column, None, None]:
        for column_index in range(self.num_columns):
            yield ColumnWiseTable.Column(self, column_index)

    @override
    def to_row_wise_table(self) -> RowWiseTable:
        table = RowWiseTable(*self._headers, alignments=self._alignments)

        for row in self.rows:
            table.add_row(*row)

        return table

    @override
    def to_column_wise_table(self) -> 'ColumnWiseTable':
        return self
