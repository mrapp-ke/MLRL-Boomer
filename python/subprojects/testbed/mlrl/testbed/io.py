"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for writing and reading files.
"""
import xml.etree.ElementTree as XmlTree

from csv import QUOTE_MINIMAL, DictReader, DictWriter
from os import listdir, path, unlink
from typing import Optional
from xml.dom import minidom

# The delimiter used to separate the columns in a CSV file
CSV_DELIMITER = ','

# The character used for quotations in a CSV file
CSV_QUOTE_CHAR = '"'

# The suffix of a text file
SUFFIX_TEXT = 'txt'

# The suffix of a CSV file
SUFFIX_CSV = 'csv'

# The suffix of an ARFF file
SUFFIX_ARFF = 'arff'

# The suffix of an XML file
SUFFIX_XML = 'xml'

# UTF-8 encoding
ENCODING_UTF8 = 'utf-8'


def get_file_name(name: str, suffix: str) -> str:
    """
    Returns a file name, including a suffix.

    :param name:    The name of the file (without suffix)
    :param suffix:  The suffix of the file
    :return:        The file name
    """
    return name + '.' + suffix


def get_file_name_per_fold(name: str, suffix: str, fold: Optional[int]) -> str:
    """
    Returns a file name, including a suffix, that corresponds to a certain fold.

    :param name:    The name of the file (without suffix)
    :param suffix:  The suffix of the file
    :param fold:    The cross validation fold, the file corresponds to, or None, if the file does not correspond to a
                    specific fold
    :return:        The file name
    """
    return get_file_name(name + '_' + ('overall' if fold is None else 'fold-' + str(fold + 1)), suffix)


def open_writable_txt_file(directory: str, file_name: str, fold: Optional[int] = None, append: bool = False):
    """
    Opens a text file to be written to.

    :param directory:   The directory where the file is located
    :param file_name:   The name of the file to be opened (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if the file does not correspond to
                        a specific fold
    :param append:      True, if new data should be appended to the file, if it already exists, False otherwise
    :return:            The file that has been opened
    """
    file = path.join(directory, get_file_name_per_fold(file_name, SUFFIX_TEXT, fold))
    write_mode = 'a' if append and path.isfile(file) else 'w'
    return open(file, mode=write_mode, encoding=ENCODING_UTF8)


def open_readable_csv_file(file_path: str):
    """
    Opens a CSV file to be read from.

    :param file_path:   The path of the file to be opened
    :return:            The file that has been opened
    """
    return open(file_path, mode='r', newline='', encoding=ENCODING_UTF8)


def open_writable_csv_file(file_path: str, append: bool = False):
    """
    Opens a CSV file to be written to.

    :param file_path:   The path of the file to be opened
    :param append:      True, if new data should be appended to the file, if it already exists, False otherwise
    :return:            The file that has been opened
    """
    write_mode = 'a' if append and path.isfile(file_path) else 'w'
    return open(file_path, mode=write_mode, encoding=ENCODING_UTF8)


def create_csv_dict_reader(csv_file) -> DictReader:
    """
    Creates and return a `DictReader` that allows to read from a CSV file.

    :param csv_file:    The CSV file
    :return:            The 'DictReader' that has been created
    """
    return DictReader(csv_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR)


def create_csv_dict_writer(csv_file, header) -> DictWriter:
    """
    Creates and returns a `DictWriter` that allows to write a dictionary to a CSV file.

    :param csv_file:    The CSV file
    :param header:      A list that contains the headers of the CSV file. They must correspond to the keys in the
                        directory that should be written to the file
    :return:            The `DictWriter` that has been created
    """
    csv_writer = DictWriter(csv_file,
                            delimiter=CSV_DELIMITER,
                            quotechar=CSV_QUOTE_CHAR,
                            quoting=QUOTE_MINIMAL,
                            fieldnames=header)

    if csv_file.mode == 'w':
        csv_writer.writeheader()

    return csv_writer


def write_xml_file(xml_file, root_element: XmlTree.Element):
    """
    Writes an XML structure to a file.

    :param xml_file:        The XML file
    :param root_element:    The root element of the XML structure
    """
    with open(xml_file, mode='w', encoding=ENCODING_UTF8) as file:
        xml_string = minidom.parseString(XmlTree.tostring(root_element)).toprettyxml(encoding=ENCODING_UTF8)
        file.write(xml_string.decode(ENCODING_UTF8))


def clear_directory(directory: str):
    """
    Deletes all files contained in a directory (excluding subdirectories).

    :param directory: The directory to be cleared
    """
    for file in listdir(directory):
        file_path = path.join(directory, file)

        if path.isfile(file_path):
            unlink(file_path)
