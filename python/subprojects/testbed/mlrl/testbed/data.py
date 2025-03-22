"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for loading and saving data sets.
"""
import logging as log
import xml.etree.ElementTree as XmlTree

from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce
from os import path
from typing import Any, List, Optional, Set, Tuple
from xml.dom import minidom

import arff
import numpy as np

from scipy.sparse import coo_array, csc_array, dok_array, lil_array
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from mlrl.common.data.arrays import is_sparse
from mlrl.common.data.types import Float32, Uint8

from mlrl.testbed.io import ENCODING_UTF8, write_xml_file


class AttributeType(Enum):
    """
    All supported types of attributes.
    """
    NUMERICAL = auto()
    ORDINAL = auto()
    NOMINAL = auto()


@dataclass
class Attribute:
    """
    An attribute, e.g., a feature, a ground truth label, or a regression score, that is contained by a data set.

    Attributes:
        name:           The name of the attribute
        attribute_type: The type of the attribute
        nominal_values: A list that contains the possible values in case of a nominal feature
    """
    name: str
    attribute_type: AttributeType
    nominal_values: Optional[List[str]] = None


@dataclass
class MetaData:
    """
    Stores the meta-data of a data set.

    Attributes:
        features:           A list that contains all features in the data set
        outputs:            A list that contains all outputs in the data set
        outputs_at_start:   True, if the outputs are located at the start, False, if they are located at the end
    """
    features: List[Attribute]
    outputs: List[Attribute]
    outputs_at_start: bool = False

    def get_num_features(self, *feature_types: AttributeType) -> int:
        """
        Returns the number of features with one out of a given set of types.  If no types are given, all features are
        counted.

        :param feature_types:   The types of the features to be counted
        :return:                The number of features of the given types
        """
        feature_types = set(feature_types)

        if feature_types:
            return reduce(lambda aggr, feature: aggr + (1 if feature.attribute_type in feature_types else 0),
                          self.features, 0)

        return len(self.features)

    def get_feature_indices(self, *feature_types: AttributeType) -> List[int]:
        """
        Returns a list that contains the indices of all features with one out of a given set of types (in ascending
        order). If no types are given, all indices are returned.

        :param feature_types:   The types of the features whose indices should be returned
        :return:                A list that contains the indices of all features of the given types
        """
        feature_types = set(feature_types)
        return [
            i for i, feature in enumerate(self.features) if not feature_types or feature.attribute_type in feature_types
        ]


def load_data_set_and_meta_data(data_dir: str,
                                arff_file_name: str,
                                xml_file_name: str,
                                feature_dtype=Float32,
                                output_dtype=Uint8) -> Tuple[lil_array, lil_array, MetaData]:
    """
    Loads a data set from an ARFF file and the corresponding Mulan XML file.

    :param data_dir:        The path to the directory that contains the files
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param xml_file_name:   The name of the XML file (including the suffix)
    :param feature_dtype:   The requested type of the feature matrix
    :param output_dtype:    The requested type of the output matrix
    :return:                A `scipy.sparse.lil_array` of type `feature_dtype`, shape `(num_examples, num_features)`,
                            representing the feature values of the examples, a `scipy.sparse.lil_array` of type
                            `output_dtype`, shape `(num_examples, num_outputs)`, representing the corresponding ground
                            truth, as well as the data set's meta-data
    """
    arff_file = path.join(data_dir, arff_file_name)
    log.debug('Loading data set from file \"%s\"...', arff_file)
    matrix, arff_attributes, relation = __load_arff(arff_file, feature_dtype=feature_dtype)
    attributes = __parse_arff_attributes(arff_attributes)
    output_names = __parse_output_names_from_xml_file(path.join(data_dir, xml_file_name))

    if not output_names:
        output_names = __parse_output_names_from_relation(relation, attributes)

    meta_data = __create_meta_data(attributes, output_names)
    x, y = __create_feature_and_output_matrix(matrix, meta_data, output_dtype)
    return x, y, meta_data


def load_data_set(data_dir: str,
                  arff_file_name: str,
                  meta_data: MetaData,
                  feature_dtype=Float32,
                  output_dtype=Uint8) -> Tuple[lil_array, lil_array]:
    """
    Loads a data set from an ARFF file given its meta-data.

    :param data_dir:        The path to the directory that contains the ARFF file
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param meta_data:       The meta-data
    :param feature_dtype:   The requested data type of the feature matrix
    :param output_dtype:    The requested data type of the output matrix
    :return:                A `scipy.sparse.lil_array` of type `feature_dtype`, shape `(num_examples, num_features)`,
                            representing the feature values of the examples, as well as a `scipy.sparse.lil_array` of
                            type `output_dtype`, shape `(num_examples, num_outputs)`, representing the corresponding
                            ground truth
    """
    arff_file = path.join(data_dir, arff_file_name)
    log.debug('Loading data set from file \"%s\"...', arff_file)
    matrix, _, _ = __load_arff(arff_file, feature_dtype=feature_dtype)
    x, y = __create_feature_and_output_matrix(matrix, meta_data, output_dtype)
    return x, y


def save_arff_file(output_dir: str, arff_file_name: str, x: np.ndarray, y: np.ndarray, meta_data: MetaData):
    """
    Saves a data set to an ARFF file.

    :param output_dir:      The path to the directory where the ARFF file should be saved
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param x:               A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                            `(num_examples, num_features)`, that stores the features of the examples that are contained
                            in the data set
    :param y:               A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                            `(num_examples, num_outputs)`, that stores the outputs of the examples that are contained in
                            the data set
    :param meta_data:       The meta-data of the data set that should be saved
    """
    arff_file = path.join(output_dir, arff_file_name)
    log.debug('Saving data set to file \'%s\'...', str(arff_file))
    sparse = is_sparse(x) and is_sparse(y)
    x = dok_array(x)
    y = dok_array(y)
    x_prefix = 0
    y_prefix = 0

    features = meta_data.features
    x_features = [(features[i].name, 'NUMERIC' if features[i].attribute_type == AttributeType.NUMERICAL
                   or features[i].nominal_values is None else features[i].nominal_values) for i in range(x.shape[1])]

    outputs = meta_data.outputs
    y_features = [(outputs[i].name, 'NUMERIC' if outputs[i].attribute_type == AttributeType.NUMERICAL
                   or outputs[i].nominal_values is None else outputs[i].nominal_values) for i in range(y.shape[1])]

    if meta_data.outputs_at_start:
        x_prefix = y.shape[1]
        relation_sign = 1
        attributes = y_features + x_features
    else:
        y_prefix = x.shape[1]
        relation_sign = -1
        attributes = x_features + y_features

    if sparse:
        data = [{} for _ in range(x.shape[0])]
    else:
        data = [[0 for _ in range(x.shape[1] + y.shape[1])] for _ in range(x.shape[0])]

    for keys, value in list(x.items()):
        data[keys[0]][x_prefix + keys[1]] = value

    for keys, value in list(y.items()):
        data[keys[0]][y_prefix + keys[1]] = value

    with open(arff_file, 'w', encoding=ENCODING_UTF8) as file:
        file.write(
            arff.dumps({
                'description': 'traindata',
                'relation': 'traindata: -C ' + str(y.shape[1] * relation_sign),
                'attributes': attributes,
                'data': data
            }))
    log.info('Successfully saved data set to file \'%s\'.', str(arff_file))


def one_hot_encode(x, y, meta_data: MetaData, encoder=None):
    """
    One-hot encodes the nominal features contained in a data set, if any.

    If the given feature matrix is sparse, it will be converted into a dense matrix. Also, an updated variant of the
    given meta-data, where the features have been removed, will be returned, as the original features become invalid by
    applying one-hot-encoding.

    :param x:           A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                        `(num_examples, num_features)`, representing the features of the examples in the data set
    :param y:           A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                        `(num_examples, num_outputs)`, representing the outputs of the examples in the data set
    :param meta_data:   The meta-data of the data set
    :param encoder:     The 'ColumnTransformer' to be used or None, if a new encoder should be created
    :return:            A `np.ndarray`, shape `(num_examples, num_encoded_features)`, representing the encoded features
                        of the given examples, the encoder that has been used, as well as the updated meta-data
    """
    nominal_indices = meta_data.get_feature_indices(AttributeType.NOMINAL)
    num_nominal_features = len(nominal_indices)
    log.info('Data set contains %s nominal and %s numerical features.', num_nominal_features,
             (len(meta_data.features) - num_nominal_features))

    if num_nominal_features > 0:
        if is_sparse(x):
            x = x.toarray()

        old_shape = x.shape

        if not encoder:
            log.info('Applying one-hot encoding...')
            encoder = ColumnTransformer(
                [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_indices)],
                remainder='passthrough')
            encoder.fit(x, y)

        x = encoder.transform(x)
        new_shape = x.shape
        updated_meta_data = MetaData([], meta_data.outputs, meta_data.outputs_at_start)
        log.info('Original data set contained %s features, one-hot encoded data set contains %s features', old_shape[1],
                 new_shape[1])
        return x, encoder, updated_meta_data

    log.debug('No need to apply one-hot encoding, as the data set does not contain any nominal features.')
    return x, None, meta_data


def __create_feature_and_output_matrix(matrix: csc_array, meta_data: MetaData,
                                       output_dtype) -> Tuple[lil_array, lil_array]:
    """
    Creates and returns the feature and output matrix from a single matrix, representing the values in an ARFF file.

    :param matrix:          A `scipy.sparse.csc_array` of type `feature_dtype`, shape
                            `(num_examples, num_features + num_outputs)`, representing the values in an ARFF file
    :param meta_data:       The meta-data of the data set
    :param output_dtype:    The requested type of the output matrix
    :return:                A `scipy.sparse.lil_array` of type `feature_dtype`, shape `(num_examples, num_features)`,
                            representing the feature matrix, as well as `scipy.sparse.lil_array` of type `output_dtype`,
                            shape `(num_examples, num_outputs)`, representing the output matrix
    """
    num_outputs = len(meta_data.outputs)

    if meta_data.outputs_at_start:
        x = matrix[:, num_outputs:]
        y = matrix[:, :num_outputs]
    else:
        x = matrix[:, :-num_outputs]
        y = matrix[:, -num_outputs:]

    x = x.tolil()
    y = y.astype(output_dtype).tolil()
    return x, y


def __load_arff(arff_file: str, feature_dtype) -> Tuple[csc_array, list, str]:
    """
    Loads the content of an ARFF file.

    :param arff_file:       The path to the ARFF file (including the suffix)
    :param feature_dtype:   The type, the data should be converted to
    :return:                A `np.sparse.csc_array` of type `feature_dtype`, containing the values in the ARFF file, a
                            list that contains a description of each feature in the ARFF file, as well as its @relation
                            name
    """
    try:
        arff_dict = __load_arff_as_dict(arff_file, sparse=True)
        data = arff_dict['data']
        matrix_data = data[0]
        matrix_row_indices = data[1]
        matrix_col_indices = data[2]
        shape = (max(matrix_row_indices) + 1, max(matrix_col_indices) + 1)
        matrix = coo_array((matrix_data, (matrix_row_indices, matrix_col_indices)), shape=shape, dtype=feature_dtype)
        matrix = matrix.tocsc()
    except arff.BadLayout:
        arff_dict = __load_arff_as_dict(arff_file, sparse=False)
        data = arff_dict['data']
        matrix = csc_array(data, dtype=feature_dtype)

    features = arff_dict['attributes']
    relation = arff_dict['relation']
    return matrix, features, relation


def __load_arff_as_dict(arff_file: str, sparse: bool) -> dict:
    """
    Loads the content of an ARFF file.

    :param arff_file:   The path to the ARFF file (including the suffix)
    :param sparse:      True, if the ARFF file is given in sparse format, False otherwise. If the given format is
                        incorrect, a `arff.BadLayout` will be raised
    :return:            A dictionary that stores the content of the ARFF file
    """
    with open(arff_file, 'r', encoding=ENCODING_UTF8) as file:
        sparse_format = arff.COO if sparse else arff.DENSE
        return arff.load(file, encode_nominal=True, return_type=sparse_format)


def __parse_arff_attributes(arff_attributes: List[Any]) -> List[Attribute]:
    """
    Parses the attributes contained in an ARFF file.

    :return:    A list that contains the attributes in the ARFF file
    """
    attributes = []

    for attribute in arff_attributes:
        attribute_name = __parse_attribute_name(attribute[0])
        type_definition = attribute[1]

        if isinstance(type_definition, list):
            feature_type = AttributeType.NOMINAL
            nominal_values = type_definition
        else:
            type_definition = str(type_definition).lower()
            nominal_values = None

            if type_definition == 'integer':
                feature_type = AttributeType.ORDINAL
            elif type_definition in ('real', 'numeric'):
                feature_type = AttributeType.NUMERICAL
            else:
                raise ValueError('Encountered unsupported feature type: ' + type_definition)

        attributes.append(Attribute(attribute_name, feature_type, nominal_values))

    return attributes


def __parse_output_names_from_xml_file(xml_file: str) -> Optional[Set[str]]:
    """
    Parses a Mulan XML file to retrieve the names of the outputs contained in a data set.

    :param xml_file:    The path to the XML file (including the suffix)
    :return:            A set that contains the names of the outputs or None, if the XML file does not exist
    """
    if path.isfile(xml_file):
        log.debug('Parsing meta-data from file \"%s\"...', xml_file)
        xml_doc = minidom.parse(xml_file)
        tags = xml_doc.getElementsByTagName('label')
        return {__parse_attribute_name(tag.getAttribute('name')) for tag in tags}

    log.debug(
        'Mulan XML file \"%s\" does not exist. If possible, information about the data set\'s outputs is parsed from '
        + 'the ARFF file\'s @relation declaration as intended by the MEKA data set format...', xml_file)
    return None


def __parse_output_names_from_relation(relation: str, attributes: List[Attribute]) -> Set[str]:
    """
    Parses the @relation declaration of an ARFF file to retrieve the names of the outputs contained in a data set.


    :param relation:    The @relation declaration to be parsed
    :param attributes:  A list that contains all attributes that are contained in the ARFF file, including features and
                        outputs
    :return:            A list that contains the names of the outputs
    """
    parameter_name = '-C '
    index = relation.index(parameter_name)
    parameter_value = relation[index + len(parameter_name):]
    index = parameter_value.find(' ')

    if index >= 0:
        parameter_value = parameter_value[:index]

    num_outputs = int(parameter_value)
    return {__parse_attribute_name(attributes[i].name) for i in range(num_outputs)}


def __create_meta_data(attributes: List[Attribute], output_names: Set[str]) -> MetaData:
    """
    Creates and returns the `MetaData` of a data set.

    :param attributes:      A list that contains all attributes in the dataset, including features and outputs
    :param output_names:    A set that contains the names of all outputs
    :return:                The `MetaData` that has been created
    """
    outputs_at_start = False
    features = []
    outputs = []

    for attribute in attributes:
        if attribute.name in output_names:
            outputs.append(attribute)

            if not features:
                outputs_at_start = True
        else:
            features.append(attribute)

    return MetaData(features, outputs, outputs_at_start)


def __parse_attribute_name(name: str) -> str:
    """
    Parses the name of an attribute and removes forbidden characters.

    :param name:    The name of the attribute
    :return:        The parsed name
    """
    name = name.strip()
    if name.startswith('\'') or name.startswith('\"'):
        name = name[1:]
    if name.endswith('\'') or name.endswith('\"'):
        name = name[:(len(name) - 1)]
    return name.replace('\\\'', '\'').replace('\\\"', '\"')


def __write_meta_data(xml_file: str, meta_data: MetaData):
    """
    Writes meta-data to a Mulan XML file.

    :param xml_file:    The path fo the XML file (including the suffix)
    :param meta_data:   The meta-data to be written
    """
    root_element = XmlTree.Element('labels')
    root_element.set('xmlns', 'http://mulan.sourceforge.net/labels')

    for output in meta_data.outputs:
        label_element = XmlTree.SubElement(root_element, 'label')
        label_element.set('name', output.name)

    write_xml_file(xml_file, root_element)
