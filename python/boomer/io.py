#!/usr/bin/python

"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for writing and reading files.
"""
import os
import os.path as path
from csv import writer, DictWriter, QUOTE_MINIMAL

# The delimiter used to separate the columns in a CSV file
CSV_DELIMITER = ','

# The character used for quotations in a CSV file
CSV_QUOTE_CHAR = '"'


def open_writable_csv_file(output_dir: str, file_name: str, fold: int, append: bool = False):
    """
    Opens a CSV file to be written to.

    :param output_dir:  The directory where the file is located
    :param file_name:   The name of the file to be opened (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if no cross validation is used
    :param append:      True, if new data should be appended to the file, if it already exists, False otherwise
    :return:            The file that has been opened
    """
    output_file_name = file_name + '_' + ('overall' if fold is None else 'fold_' + str(fold + 1)) + '.csv'
    output_file = path.join(output_dir, output_file_name)
    write_mode = 'a' if append and path.isfile(output_file) else 'w'
    return open(output_file, mode=write_mode)


def create_csv_writer(csv_file):
    """
    Creates and returns a writer that allows to write rows to a CSV file.

    :param csv_file:    The CSV file
    :return:            The writer that has been created
    """
    return writer(csv_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR, quoting=QUOTE_MINIMAL)


def create_csv_dict_writer(csv_file, header) -> DictWriter:
    """
    Creates and returns a `DictWriter` that allows to write a dictionary to a CSV file.

    :param csv_file:    The CSV file
    :param header:      A list that contains the headers of the CSV file. They must correspond to the keys in the
                        directory that should be written to the file
    :return:            The `DictWriter` that has been created
    """
    csv_writer = DictWriter(csv_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR, quoting=QUOTE_MINIMAL,
                            fieldnames=header)

    if csv_file.mode == 'w':
        csv_writer.writeheader()

    return csv_writer


def clear_directory(directory: str):
    """
    Deletes all files contained in a directory (excluding subdirectories).

    :param directory: The directory to be cleared
    """
    for file in os.listdir(directory):
        file_path = path.join(directory, file)

        if path.isfile(file_path):
            os.unlink(file_path)
