"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for loading and saving data sets.
"""
import logging as log

from dataclasses import dataclass
from os import path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.dom import minidom

import arff

from scipy.sparse import coo_array, csc_array, lil_array

from mlrl.common.data.types import Float32, Uint8

from mlrl.testbed.dataset import Attribute, AttributeType
from mlrl.testbed.util.io import ENCODING_UTF8


@dataclass
class ArffMetaData:
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


def load_data_set_and_meta_data(directory: str,
                                arff_file_name: str,
                                xml_file_name: Optional[str],
                                feature_dtype=Float32,
                                output_dtype=Uint8) -> Tuple[lil_array, lil_array, ArffMetaData]:
    """
    Loads a data set from an ARFF file and the corresponding Mulan XML file.

    :param directory:       The path to the directory that contains the files
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param xml_file_name:   The name of the XML file (including the suffix), if available
    :param feature_dtype:   The requested type of the feature matrix
    :param output_dtype:    The requested type of the output matrix
    :return:                A `scipy.sparse.lil_array` of type `feature_dtype`, shape `(num_examples, num_features)`,
                            representing the feature values of the examples, a `scipy.sparse.lil_array` of type
                            `output_dtype`, shape `(num_examples, num_outputs)`, representing the corresponding ground
                            truth, as well as the data set's meta-data
    """
    arff_file = path.join(directory, arff_file_name)
    log.debug('Loading data set from file \"%s\"...', arff_file)
    matrix, arff_attributes, relation = __load_arff(arff_file, feature_dtype=feature_dtype)
    attributes = __parse_arff_attributes(arff_attributes)
    output_names = __parse_output_names_from_xml_file(path.join(directory, xml_file_name))

    if not output_names:
        output_names = __parse_output_names_from_relation(relation, attributes)

    meta_data = __create_meta_data(attributes, output_names)
    x, y = __create_feature_and_output_matrix(matrix, meta_data, output_dtype)
    return x, y, meta_data


def load_data_set(directory: str,
                  arff_file_name: str,
                  meta_data: ArffMetaData,
                  feature_dtype=Float32,
                  output_dtype=Uint8) -> Tuple[lil_array, lil_array]:
    """
    Loads a data set from an ARFF file given its meta-data.

    :param directory:       The path to the directory that contains the ARFF file
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param meta_data:       The meta-data
    :param feature_dtype:   The requested data type of the feature matrix
    :param output_dtype:    The requested data type of the output matrix
    :return:                A `scipy.sparse.lil_array` of type `feature_dtype`, shape `(num_examples, num_features)`,
                            representing the feature values of the examples, as well as a `scipy.sparse.lil_array` of
                            type `output_dtype`, shape `(num_examples, num_outputs)`, representing the corresponding
                            ground truth
    """
    arff_file = path.join(directory, arff_file_name)
    log.debug('Loading data set from file \"%s\"...', arff_file)
    matrix, _, _ = __load_arff(arff_file, feature_dtype=feature_dtype)
    x, y = __create_feature_and_output_matrix(matrix, meta_data, output_dtype)
    return x, y


def __create_feature_and_output_matrix(matrix: csc_array, meta_data: ArffMetaData,
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


def __load_arff_as_dict(arff_file: str, sparse: bool) -> Dict[str, Any]:
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


def __create_meta_data(attributes: List[Attribute], output_names: Set[str]) -> ArffMetaData:
    """
    Creates and returns the `ArffMetaData` of a data set.

    :param attributes:      A list that contains all attributes in the dataset, including features and outputs
    :param output_names:    A set that contains the names of all outputs
    :return:                The `ArffMetaData` that has been created
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

    return ArffMetaData(features, outputs, outputs_at_start)


def __parse_attribute_name(name: str) -> str:
    """
    Parses the name of an attribute and removes forbidden characters.

    :param name:    The name of the attribute
    :return:        The parsed name
    """
    name = name.strip()
    if name.startswith('\'') or name.startswith('"'):
        name = name[1:]
    if name.endswith('\'') or name.endswith('"'):
        name = name[:(len(name) - 1)]
    return name.replace('\\\'', '\'').replace('\\"', '"')
