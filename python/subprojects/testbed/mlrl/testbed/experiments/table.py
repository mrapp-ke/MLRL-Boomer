"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing tables.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Iterator
from enum import Enum
from typing import Any, override, Sequence
from rich.table import Table as RichTable
from rich import box
from rich.text import Text
from rich.console import ConsoleRenderable
from rich.style import Style


Header = Any | None

Cell = Any | None


class HeaderRow(Iterable[Header], ABC):
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

    def __len__(self) -> int:
        """
        Returns the number of cells in the row.

        :return: The number of cells in the row
        """
        return self.num_columns


class Column(Iterable[Cell], ABC):
    """
    Provides access to a single column of a table.
    """

    class Alignment(Enum):
        """
        All possible alignments of a table column.
        """

        AUTO = None
        LEFT = 'left'
        CENTER = 'center'
        RIGHT = 'right'

    class Style(Enum):
        """
        Different styles of a column.
        """

        HEADER = Style(bold=True)
        VALUE = Style(color='turquoise2')
        VALUE_SECONDARY = Style(color='turquoise4')

    @abstractmethod
    def set_header(self, value: Header):
        """
        Sets the header of the column to a given value.

        :param value: The value to be set
        """

    @property
    @abstractmethod
    def header(self) -> Header | None:
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

    class BorderStyle(Enum):
        """
        All border styles that can be used by tables.
        """

        NONE = None
        HORIZONTAL_LINES = box.HORIZONTALS
        INNER_LINES = box.MINIMAL
        ALL_LINES = box.ROUNDED

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
    def header_row(self) -> HeaderRow | None:
        """
        The header row of the table or None, if the table does not have any headers.
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

    def to_rich_table(
        self,
        auto_rotate: bool = True,
        border_style: 'Table.BorderStyle' = BorderStyle.NONE,
        column_styles: Sequence[Column.Style | None] | None = None,
        column_alignments: Sequence[Column.Alignment | None] | None = None,
        separator_indices: Sequence[int] | None = None,
    ) -> ConsoleRenderable:
        """
        Creates and returns an object of type `rich.table.Table`.

        :param auto_rotate:         True, if tables with a single row should automatically be rotated for legibility,
                                    False otherwise
        :param border_style:        The border style to be used for formatting the table
        :param column_styles:       A sequence that contains the style of the columns of None, if no styling should be
                                    used
        :param column_alignments:   A sequence that contains the alignments of individual columns or None, if the
                                    default should be used
        :param separator_indices:   A sequence that contains the indices of the row at which a separator should be
                                    inserted, or None if no separators should be inserted
        :return:                    The object that has been created
        """
        if self.num_cells > 0:
            headers = self.header_row
            rows: Iterable[Any] = self.rows
            alignments: list[Column.Alignment | None] = list(column_alignments) if column_alignments else []
            styles: list[Column.Style | None] = list(column_styles) if column_styles else []

            if auto_rotate and headers and self.num_rows == 1:
                first_row = next(self.rows)
                rows = [[headers[column_index], first_row[column_index]] for column_index in range(self.num_columns)]
                alignments = [Column.Alignment.LEFT, Column.Alignment.RIGHT]
                styles = [Column.Style.HEADER, Column.Style.VALUE]
                headers = None

            rich_table = RichTable(
                show_header=bool(headers),
                box=border_style.value if headers else None,
                header_style=Column.Style.HEADER.value,
            )
            num_columns = max(len(alignments), len(styles), headers.num_columns if headers else 0)

            for column_index in range(num_columns):
                alignment = alignments[column_index] if len(alignments) > column_index else None
                column_style = styles[column_index] if len(styles) > column_index else None
                style = column_style.value if column_style else None
                header = headers[column_index] if headers and headers.num_columns > column_index else None
                justify = alignment.value if alignment else None

                if header:
                    rich_table.add_column(str(header), style=style, justify=justify)  # type: ignore[arg-type]
                else:
                    rich_table.add_column(style=style, justify=justify)  # type: ignore[arg-type]

            indices = set(separator_indices) if separator_indices else set()

            for row_index, row in enumerate(rows):
                rich_table.add_row(
                    *(
                        str(row[column_index]) if len(row) > column_index and row[column_index] else None
                        for column_index in range(num_columns)
                    ),
                    end_section=row_index + 1 in indices,
                )

            return rich_table

        return Text('<no data available>')


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
            headers = self.table._headers
            return headers[column_index] if headers else None

        @override
        def __iter__(self) -> Iterator[Header]:
            headers = self.table._headers
            return iter(headers if headers else [])

        @override
        def __eq__(self, other: Any) -> bool:
            return (
                isinstance(other, type(self))
                and other.num_columns == self.num_columns
                and all(str(first_header) == str(second_header) for first_header, second_header in zip(self, other))
            )

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
        def __iter__(self) -> Iterator[Cell]:
            for column_index in range(self.num_columns):
                yield self[column_index]

        @override
        def __getitem__(self, column_index: int) -> Cell:
            if column_index >= self.num_columns:
                raise IndexError(
                    f'Invalid column index: Got {column_index}, but table has only {self.num_columns} columns'
                )

            row = self.table._rows[self.row_index]
            return row[column_index] if column_index < len(row) else None

        def __setitem__(self, column_index: int, value: Cell):
            if column_index >= self.num_columns:
                raise IndexError(
                    f'Invalid column index: Got {column_index}, but table has only {self.num_columns} columns'
                )

            row = self.table._rows[self.row_index]

            if len(row) <= column_index:
                row.extend([None] * (column_index - len(row) + 1))

            row[column_index] = value

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
        def set_header(self, value: Header):
            headers = self.table._headers

            if headers:
                headers[self.column_index] = value
            else:
                headers = [None for _ in range(self.table.num_columns)]
                headers[self.column_index] = value
                self.table._headers = headers

        @override
        @property
        def header(self) -> Header | None:
            header_row = self.table.header_row
            return header_row[self.column_index] if header_row else None

        @override
        @property
        def num_rows(self) -> int:
            return self.table.num_rows

        @override
        def __getitem__(self, row_index: int) -> Cell:
            if row_index >= self.num_rows:
                raise IndexError(f'Invalid row index: Got {row_index}, but table has only {self.num_rows} rows')

            row = self.table._rows[row_index]
            column_index = self.column_index
            return row[column_index] if column_index < len(row) else None

        @override
        def __iter__(self) -> Iterator[Cell]:
            for row_index in range(self.num_rows):
                yield self[row_index]

    def __init__(self, *headers: Header):
        """
        :param headers: The headers of the individual columns
        """
        self._headers = list(headers) if headers else None
        self._rows: list[list[Any]] = []
        self._num_columns = len(headers) if headers else 0

    @staticmethod
    def aggregate(*tables: Table) -> 'RowWiseTable':
        """
        Creates and returns a `RowWiseTable` that contains all rows of the given tables. The given tables must all have
        the same headers.

        :param tables:  The tables to aggregate
        :return:        A `RowWiseTable` that contains all rows of the given tables
        """
        row_wise_tables = [table.to_row_wise_table() for table in tables]

        if not row_wise_tables:
            raise ValueError('No tables to aggregate')

        aggregated_table = row_wise_tables[0].copy()

        if len(row_wise_tables) > 1:
            for table in row_wise_tables[1:]:
                if aggregated_table.header_row != table.header_row:
                    raise ValueError('The headers of the tables do not match')

                for row in table.rows:
                    aggregated_table.add_row(*row)

        return aggregated_table

    def copy(self) -> 'RowWiseTable':
        """
        Creates and returns a copy of this table.

        :return: The copy that has been created
        """
        copied_table = RowWiseTable(*(self._headers if self._headers else []))

        for row in self.rows:
            copied_table.add_row(*row)

        return copied_table

    def add_row(self, *values: Cell, position: int = -1):
        """
        Adds a new row at a specific position of the table.

        :param values:      The values in the row
        :param position:    The position where the row should be added or -1, if the row should be added at the end
        :return:            The modified table
        """
        row = list(values)
        headers = self._headers

        if headers and len(row) > len(headers):
            headers.append(None)

        self._rows.insert(self.num_rows if position < 0 else position, row)
        self._num_columns = max(len(row), self._num_columns)
        return self

    def sort_by_columns(
        self, column_index: int, *additional_column_indices: int, descending: bool = False
    ) -> 'RowWiseTable':
        """
        Sorts the rows in the table by the values in one or several columns.

        :param column_index:                The index of the column to sort by
        :param additional_column_indices:   Additional indices of columns to sort by
        :param descending:                  True, if the rows should be sorted in descending order, False, if they
                                            should be sorted in ascending order
        :return:                            The sorted table
        """
        self._rows.sort(
            key=lambda row: ([row[i] for i in [column_index] + list(additional_column_indices)]), reverse=descending
        )
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
    def header_row(self) -> HeaderRow | None:
        return RowWiseTable.HeaderRow(self) if self._headers else None

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

        for column_index, column in enumerate(self.columns):
            table.add_column(*column, header=header_row[column_index] if header_row else None)

        return table

    def __getitem__(self, row_index: int) -> Row:
        if row_index < 0 or row_index >= self.num_rows:
            raise ValueError(f'Row index must be between 0 and {self.num_rows}, but got: {row_index}')
        return RowWiseTable.Row(self, row_index)


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
            headers = self.table._headers
            return headers[column_index] if headers else None

        @override
        def __iter__(self) -> Iterator[Header]:
            headers = self.table._headers
            return iter(headers if headers else [])

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
                raise IndexError(
                    f'Invalid column index: Got {column_index}, but table has only {self.num_columns} columns'
                )

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
        def set_header(self, value: Header):
            """
            Sets the header of the column to a given value.

            :param value: The value to be set
            """
            headers = self.table._headers

            if headers:
                headers[self.column_index] = value
            else:
                headers = [None for _ in range(self.table.num_columns)]
                headers[self.column_index] = value
                self.table._headers = headers

        @override
        @property
        def header(self) -> Header | None:
            header_row = self.table.header_row
            return header_row[self.column_index] if header_row else None

        @override
        @property
        def num_rows(self) -> int:
            return self.table.num_rows

        @override
        def __iter__(self) -> Iterator[Cell]:
            for row_index in range(self.num_rows):
                yield self[row_index]

        @override
        def __getitem__(self, row_index: int) -> Cell:
            if row_index >= self.num_rows:
                raise IndexError(f'Invalid row index: Got {row_index}, but table has only {self.num_rows} rows')

            column = self.table._columns[self.column_index]
            return column[row_index] if row_index < len(column) else None

        def __setitem__(self, row_index: int, value: Cell):
            if row_index >= self.num_rows:
                raise IndexError(f'Invalid row index: Got {row_index}, but table has only {self.num_rows} rows')

            column = self.table._columns[self.column_index]

            if len(column) <= row_index:
                column.extend([None] * (row_index - len(column) + 1))

            column[row_index] = value

    def __init__(self):
        self._headers: list[Header] | None = None
        self._columns: list[list[Cell]] = []
        self._num_rows = 0

    def slice(self, *column_indices: int) -> 'ColumnWiseTable':
        """
        Creates and returns a copy of this table that contains the columns with specific indices in the given order.

        :param column_indices:  The indices of the columns to be sliced
        :return:                The table that has been created
        """
        sliced_table = ColumnWiseTable()

        for column_index in column_indices:
            column = self[column_index]
            sliced_table.add_column(*column, header=column.header)

        return sliced_table

    def add_column(self, *values: Cell, header: Header | None = None, position: int = -1) -> 'ColumnWiseTable':
        """
        Adds a new column to a table at a specific position.

        :param values:      The values in the column
        :param header:      The header of the column or None
        :param position:    The position, the column should be added at, or -1, if the column should be added at the end
        :return:            The modified table
        """
        column = list(values)
        position = self.num_columns if position < 0 else position

        if header:
            if not self._headers:
                self._headers = [None for _ in range(self.num_columns)]

            self._headers.insert(position, header)

        self._columns.insert(position, column)
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
            sorted_column_indices = [
                index for index, _ in sorted(enumerate(headers), key=lambda x: str(x[1]) if x[1] else '')
            ]
            sorted_columns = []
            sorted_headers = []

            for column_index in sorted_column_indices:
                sorted_columns.append(columns[column_index])
                sorted_headers.append(headers[column_index])

            self._headers = sorted_headers
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
    def header_row(self) -> HeaderRow | None:
        return ColumnWiseTable.HeaderRow(self) if self._headers else None

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
        headers = self._headers if self._headers else []
        table = RowWiseTable(*headers)

        for row in self.rows:
            table.add_row(*row)

        return table

    @override
    def to_column_wise_table(self) -> 'ColumnWiseTable':
        return self

    def __getitem__(self, column_index: int) -> Column:
        if column_index < 0 or column_index >= self.num_columns:
            raise ValueError(f'Column index must be between 0 and {self.num_columns}, but got: {column_index}')
        return ColumnWiseTable.Column(self, column_index)
