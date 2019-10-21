#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for handling multi-label data.
"""
import logging as log
import os.path as path
from typing import List, Set
from xml.dom import minidom

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from skmultilearn.dataset import load_from_arff


class Attribute:
    """
    Represents a numeric or nominal attribute contained in a data set.
    """

    def __init__(self, name: str):
        """
        :param name:    The name of the attribute
        """
        self.name = name.strip()
        if self.name.startswith('\'') or self.name.startswith('\"'):
            self.name = self.name[1:]
        if self.name.endswith('\'') or self.name.endswith('\"'):
            self.name = self.name[:(len(self.name) - 1)]
        self.name = self.name.replace('\\\'', '\'').replace('\\\"', '\"')


class NominalAttribute(Attribute):
    """
    Represents a nominal attribute contained in a data set.
    """

    def __init__(self, name):
        """
        :param name:    The name of the attribute
        """
        super().__init__(name)


class MetaData:
    """
    Stores the meta data of a multi-label data set.
    """

    def __init__(self, num_labels: int, label_location: str, nominal_attributes: List[int]):
        """
        :param num_labels:          The number of labels contained in the data set
        :param label_location:      Whether the labels are located at the 'start', or the 'end'
        :param nominal_attributes:  A list that contains the indices of all nominal attributes
        """
        self.num_labels = num_labels
        self.label_location = label_location
        self.nominal_attributes = nominal_attributes


def load_data_set_and_meta_data(data_dir: str, arff_file_name: str, xml_file_name: str) -> (np.ndarray, np.ndarray, MetaData):
    """
    Loads a multi-label data set from an ARFF file and the corresponding Mulan XML file..

    :param data_dir:        The path of the directory that contains the files
    :param arff_file_name:  The name of the ARFF file
    :param xml_file_name:   The name of the XML file
    :return:                An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                            the examples, an array of dtype float, shape `(num_examples, num_labels)`, representing the
                            corresponding label vectors, as well as the data set's meta data
    """

    arff_file = path.join(data_dir, arff_file_name)
    xml_file = path.join(data_dir, xml_file_name)
    log.debug('Parsing meta data from file \"%s\"...', xml_file)
    meta_data = __parse_meta_data(arff_file, xml_file)
    x, y = load_data_set(data_dir, arff_file_name, meta_data)
    return x, y, meta_data


def load_data_set(data_dir: str, arff_file_name: str, meta_data: MetaData) -> (np.ndarray, np.ndarray):
    """
    Loads a multi-label data set from an ARFF file given its meta data.

    :param data_dir:        The path of the directory that contains the ARFF file
    :param arff_file_name:  The name of the ARFF file
    :param meta_data:       The meta data
    :return:                An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                            the examples, as well as an array of dtype float, shape `(num_examples, num_labels)`,
                            representing the corresponding label vectors
    """

    arff_file = path.join(data_dir, arff_file_name)
    log.debug('Loading data set from file \"%s\"...', arff_file)
    x, y = load_from_arff(arff_file, label_count=meta_data.num_labels, label_location=meta_data.label_location)
    return x, y


def one_hot_encode(x, y, nominal_indices, encoder=None):
    """
    One-hot encodes the nominal attributes contained in a data set, if any.

    :param x:               The features of the examples in the data set
    :param y:               The labels of the examples in the data set
    :param nominal_indices: A list containing the indices of all nominal attributes
    :param encoder:         The 'OneHotEncoder' to be used or None, if a new encoder should be created
    :return:                The encoded features of the given examples and the encoder that has been used
    """

    if len(nominal_indices) > 0:
        x = x.toarray()
        old_shape = x.shape

        if encoder is None:
            log.info('Applying one-hot encoding...')
            encoder = ColumnTransformer(
                [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False), nominal_indices)],
                remainder='passthrough')
            encoder.fit(x, y)

        x = encoder.transform(x)
        new_shape = x.shape
        log.info('Original data set contained %s attributes, one-hot encoded data set contains %s attributes',
                 old_shape[1], new_shape[1])
        return x, encoder
    else:
        log.debug('No need to apply one-hot encoding, as the data set does not contain any nominal attributes.')
        return x, None


def __parse_labels(metadata_file) -> Set[str]:
    """
    Parses a Mulan XML file to retrieve information about the labels contained in a data set.

    :param metadata_file:   The path of the XML file
    :return:                A set containing the names of the labels
    """

    xml_doc = minidom.parse(metadata_file)
    tags = xml_doc.getElementsByTagName('label')
    return set([tag.getAttribute('name') for tag in tags])


def __parse_attributes(arff_file) -> List[Attribute]:
    """
    Parses an ARFF file to retrieve information about the attributes contained in a data set.

    :param arff_file:   The path of the ARFF file
    :return:            A list containing the attributes
    """

    attributes: List[Attribute] = []

    with open(arff_file) as file:
        for line in file:
            if line.startswith('@attribute'):
                attribute_definition = line[len('@attribute'):].strip()

                if attribute_definition.endswith('numeric'):
                    # Numerical attribute
                    attribute_name = attribute_definition[:(len(attribute_definition) - len('numeric'))]
                    attributes.append(Attribute(attribute_name))
                else:
                    # Nominal attribute
                    attribute_name = attribute_definition[:attribute_definition.find(' {')]
                    attributes.append(NominalAttribute(attribute_name))

    return attributes


def __parse_meta_data(arff_file, metadata_file) -> MetaData:
    """
    Parses meta data from an ARFF file and the corresponding Mulan XML file.

    :param arff_file:       The path of the ARFF file
    :param metadata_file:   The path of the XML file
    :return:                The number of labels, the location of the labels ('start' or 'end'), the indices of all
                            nominal attributes
    """

    labels = __parse_labels(metadata_file)
    attributes = __parse_attributes(arff_file)
    labels_located_at_start = attributes[0].name in labels
    label_location = 'start' if labels_located_at_start else 'end'
    nominal_indices = [i - (len(labels) if labels_located_at_start else 0) for i, attribute in enumerate(attributes)
                       if isinstance(attribute, NominalAttribute) and attribute.name not in labels]
    log.info('Data set contains %s nominal and %s numerical attributes.', len(nominal_indices),
             (len(attributes) - len(nominal_indices)))
    return MetaData(len(labels), label_location, nominal_indices)
