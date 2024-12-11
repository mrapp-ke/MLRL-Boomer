"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for creating tables.
"""
from core.build_unit import BuildUnit
from util.pip import Pip


class Table:
    """
    A table with optional headers.
    """

    def __init__(self, build_unit: BuildUnit, *headers: str):
        """
        :param build_unit:  The build unit, the table is created for
        :param headers:     The headers of the table
        """
        self.build_unit = build_unit
        self.headers = list(headers) if headers else None
        self.rows = []

    def add_row(self, *entries: str):
        """
        Adds a new row to the end of the table.

        :param entries: The entries of the row to be added
        """
        self.rows.append(list(entries))

    def sort_rows(self, column_index: int, *additional_column_indices: int):
        """
        Sorts the rows in the table.

        :param column_index:                The index of the column to sort by
        :param additional_column_indices:   Additional indices of columns to sort by
        """
        self.rows.sort(key=lambda row: ([row[i] for i in [column_index] + list(additional_column_indices)]))

    def __str__(self) -> str:
        Pip.for_build_unit(self.build_unit).install_packages('tabulate')
        # pylint: disable=import-outside-toplevel
        from tabulate import tabulate
        return tabulate(self.rows, headers=self.headers)
