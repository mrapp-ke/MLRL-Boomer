#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for handling multi-label data.
"""
import logging as log
from typing import List, Set
from xml.dom import minidom

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


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


def parse_metadata(arff_file, metadata_file):
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
    return len(labels), label_location, nominal_indices


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
