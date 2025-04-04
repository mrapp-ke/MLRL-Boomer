"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for reading and writing files.
"""
import xml.etree.ElementTree as XmlTree

from os import listdir, path, unlink
from typing import Optional
from xml.dom import minidom

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


def open_writable_file(file_path: str, append: bool = False):
    """
    Opens a file to be written to.

    :param file_path:   The path to the file to be opened
    :param append:      True, if new data should be appended to the file, if it already exists, False otherwise
    :return:            The file that has been opened
    """
    mode = 'a' if append and path.isfile(file_path) else 'w'
    return open(file_path, mode=mode, encoding=ENCODING_UTF8)


def open_readable_file(file_path: str):
    """
    Opens a file to be read from.

    :param file_path:   The path to the file to be opened
    :return:            The file that has been opened
    """
    return open(file_path, mode='r', newline='', encoding=ENCODING_UTF8)


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
