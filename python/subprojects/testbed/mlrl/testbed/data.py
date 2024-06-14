"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for loading and saving data sets.
"""
import logging as log
import xml.etree.ElementTree as XmlTree

from enum import Enum, auto
from functools import reduce
from os import path
from typing import List, Optional, Set, Tuple
from xml.dom import minidom

import arff
import numpy as np

from scipy.sparse import coo_array, csc_array, dok_array, lil_array
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from mlrl.common.arrays import is_sparse
from mlrl.common.data_types import Float32, Uint8

from mlrl.testbed.io import ENCODING_UTF8, write_xml_file


class FeatureType(Enum):
    """
    All supported types of features.
    """
    NUMERICAL = auto()
    ORDINAL = auto()
    NOMINAL = auto()


class Feature:
    """
    Represents a numerical or nominal feature that is contained by a data set.
    """

    def __init__(self, name: str, feature_type: FeatureType, nominal_values: Optional[List[str]] = None):
        """
        :param name:            The name of the feature
        :param feature_type:    The type of the feature
        :param nominal_values:  A list that contains the possible values in case of a nominal feature, None otherwise
        """
        self.name = name
        self.feature_type = feature_type
        self.nominal_values = nominal_values


class Output(Feature):
    """
    Represents an output that is contained by a data set.
    """

    def __init__(self, name: str):
        super().__init__(name, FeatureType.NOMINAL, [str(0), str(1)])


class MetaData:
    """
    Stores the meta-data of a data set.
    """

    def __init__(self, features: List[Feature], outputs: List[Output], outputs_at_start: bool):
        """
        :param features:            A list that contains all features in the data set
        :param outputs:             A list that contains all outputs in the data set
        :param outputs_at_start:    True, if the outputs are located at the start, False, if they are located at the end
        """
        self.features = features
        self.outputs = outputs
        self.outputs_at_start = outputs_at_start

    def get_num_features(self, feature_types: Optional[Set[FeatureType]] = None) -> int:
        """
        Returns the number of features with one out of a given set of types.

        :param feature_types:   A set that contains the types of the features to be counted or None, if all features
                                should be counted
        :return:                The number of features of the given types
        """
        if feature_types is None:
            return len(self.features)
        if len(feature_types) == 0:
            return 0
        return reduce(lambda num, feature: num + (1 if feature.feature_type in feature_types else 0), self.features, 0)

    def get_feature_indices(self, feature_types: Optional[Set[FeatureType]] = None) -> List[int]:
        """
        Returns a list that contains the indices of all features with one out of a given set of types (in ascending
        order).

        :param feature_types:   A set that contains the types of the features whose indices should be returned or
                                None, if all indices should be returned
        :return:                A list that contains the indices of all features of the given types
        """
        return [
            i for i, feature in enumerate(self.features)
            if feature_types is None or feature.feature_type in feature_types
        ]


def load_data_set_and_meta_data(data_dir: str,
                                arff_file_name: str,
                                xml_file_name: str,
                                feature_dtype=Float32,
                                output_dtype=Uint8) -> Tuple[lil_array, lil_array, MetaData]:
    """
    Loads a data set from an ARFF file and the corresponding Mulan XML file.

    :param data_dir:        The path of the directory that contains the files
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param xml_file_name:   The name of the XML file (including the suffix)
    :param feature_dtype:   The requested type of the feature matrix
    :param output_dtype:    The requested type of the output matrix
    :return:                A `scipy.sparse.lil_array` of type `feature_dtype`, shape `(num_examples, num_features)`,
                            representing the feature values of the examples, a `scipy.sparse.lil_array` of type
                            `output_dtype`, shape `(num_examples, num_outputs)`, representing the corresponding ground
                            truth, as well as the data set's meta-data
    """
    xml_file = path.join(data_dir, xml_file_name)
    outputs = None

    if path.isfile(xml_file):
        log.debug('Parsing meta data from file \"%s\"...', xml_file)
        outputs = __parse_outputs_from_xml_file(xml_file)
    else:
        log.debug(
            'Mulan XML file \"%s\" does not exist. If possible, information about the data set\'s outputs is parsed '
            + 'from the ARFF file\'s @relation declaration as intended by the MEKA data set format...', xml_file)

    arff_file = path.join(data_dir, arff_file_name)
    log.debug('Loading data set from file \"%s\"...', arff_file)
    matrix, features, relation = __load_arff(arff_file, feature_dtype=feature_dtype)

    if outputs is None:
        outputs = __parse_outputs_from_relation(relation, features)

    meta_data = __create_meta_data(features, outputs)
    x, y = __create_feature_and_output_matrix(matrix, meta_data, output_dtype)
    return x, y, meta_data


def load_data_set(data_dir: str,
                  arff_file_name: str,
                  meta_data: MetaData,
                  feature_dtype=Float32,
                  output_dtype=Uint8) -> Tuple[lil_array, lil_array]:
    """
    Loads a data set from an ARFF file given its meta-data.

    :param data_dir:        The path of the directory that contains the ARFF file
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


def save_data_set_and_meta_data(output_dir: str, arff_file_name: str, xml_file_name: str, x: np.ndarray,
                                y: np.ndarray) -> MetaData:
    """
    Saves a data set to an ARFF file and its meta-data to an XML file. All features in the data set are considered to
    be numerical.

    :param output_dir:      The path of the directory where the ARFF file and the XML file should be saved
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param xml_file_name:   The name of the XML file (including the suffix)
    :param x:               An array of type `float`, shape `(num_examples, num_features)`, representing the features of
                            the examples that are contained in the data set
    :param y:               An array of type `float`, shape `(num_examples, num_outputs)`, representing the ground truth
                            of the examples that are contained in the data set
    :return:                The meta-data of the data set that has been saved
    """
    meta_data = save_data_set(output_dir, arff_file_name, x, y)
    save_meta_data(output_dir, xml_file_name, meta_data)
    return meta_data


def save_data_set(output_dir: str, arff_file_name: str, x: np.ndarray, y: np.ndarray) -> MetaData:
    """
    Saves a data set to an ARFF file. All features in the data set are considered to be numerical.

    :param output_dir:      The path of the directory where the ARFF file should be saved
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param x:               A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                            `(num_examples, num_features)`, that stores the features of the examples that are contained
                            in the data set
    :param y:               A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                            `(num_examples, num_outputs)`, that stores the outputs of the examples that are contained in
                            the data set
    :return:                The meta-data of the data set that has been saved
    """

    num_features = x.shape[1]
    features = [Feature('X' + str(i), FeatureType.NUMERICAL) for i in range(num_features)]
    num_outputs = y.shape[1]
    outputs = [Output('y' + str(i)) for i in range(num_outputs)]
    meta_data = MetaData(features, outputs, outputs_at_start=False)
    save_arff_file(output_dir, arff_file_name, x, y, meta_data)
    return meta_data


def save_arff_file(output_dir: str, arff_file_name: str, x: np.ndarray, y: np.ndarray, meta_data: MetaData):
    """
    Saves a data set to an ARFF file.

    :param output_dir:      The path of the directory where the ARFF file should be saved
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
    x_features = [(features[i].name if len(features) > i else 'X' + str(i),
                   'NUMERIC' if len(features) <= i or features[i].nominal_values is None
                   or features[i].feature_type == FeatureType.NUMERICAL else features[i].nominal_values)
                  for i in range(x.shape[1])]

    outputs = meta_data.outputs
    y_features = [(outputs[i].name if len(outputs) > i else 'y' + str(i),
                   'NUMERIC' if len(outputs) <= i or outputs[i].nominal_values is None
                   or outputs[i].feature_type == FeatureType.NUMERICAL else outputs[i].nominal_values)
                  for i in range(y.shape[1])]

    if meta_data.outputs_at_start:
        x_prefix = y.shape[1]
        relation_sign = 1
        features = y_features + x_features
    else:
        y_prefix = x.shape[1]
        relation_sign = -1
        features = x_features + y_features

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
                'attributes': features,
                'data': data
            }))
    log.info('Successfully saved data set to file \'%s\'.', str(arff_file))


def save_meta_data(output_dir: str, xml_file_name: str, meta_data: MetaData):
    """
    Saves the meta-data of a data set to an XML file.

    :param output_dir:      The path of the directory where the XML file should be saved
    :param xml_file_name:   The name of the XML file (including the suffix)
    :param meta_data:       The meta-data of the data set
    """
    xml_file = path.join(output_dir, xml_file_name)
    log.debug('Saving meta data to file \'%s\'...', str(xml_file))
    __write_meta_data(xml_file, meta_data)
    log.info('Successfully saved meta data to file \'%s\'.', str(xml_file))


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
    nominal_indices = meta_data.get_feature_indices({FeatureType.NOMINAL})
    num_nominal_features = len(nominal_indices)
    log.info('Data set contains %s nominal and %s numerical features.', num_nominal_features,
             (len(meta_data.features) - num_nominal_features))

    if num_nominal_features > 0:
        if is_sparse(x):
            x = x.toarray()

        old_shape = x.shape

        if encoder is None:
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

    :param arff_file:       The path of the ARFF file (including the suffix)
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

    :param arff_file:   The path of the ARFF file (including the suffix)
    :param sparse:      True, if the ARFF file is given in sparse format, False otherwise. If the given format is
                        incorrect, a `arff.BadLayout` will be raised
    :return:            A dictionary that stores the content of the ARFF file
    """
    with open(arff_file, 'r', encoding=ENCODING_UTF8) as file:
        sparse_format = arff.COO if sparse else arff.DENSE
        return arff.load(file, encode_nominal=True, return_type=sparse_format)


def __parse_outputs_from_xml_file(xml_file) -> List[Output]:
    """
    Parses a Mulan XML file to retrieve information about the outputs contained in a data set.

    :param xml_file:    The path of the XML file (including the suffix)
    :return:            A list containing the outputs
    """

    xml_doc = minidom.parse(xml_file)
    tags = xml_doc.getElementsByTagName('label')
    return [Output(__parse_feature_or_output_name(tag.getAttribute('name'))) for tag in tags]


def __parse_outputs_from_relation(relation: str, features: list) -> List[Output]:
    """
    Parses the @relation declaration of an ARFF file to retrieve information about the outputs contained in a data set.

    :param relation:    The @relation declaration to be parsed
    :return:            A list containing the outputs
    """
    parameter_name = '-C '
    index = relation.index(parameter_name)
    parameter_value = relation[index + len(parameter_name):]
    index = parameter_value.find(' ')

    if index >= 0:
        parameter_value = parameter_value[:index]

    num_outputs = int(parameter_value)
    return [Output(__parse_feature_or_output_name(features[i][0])) for i in range(num_outputs)]


def __create_meta_data(features: List[Feature], outputs: List[Output]) -> MetaData:
    """
    Creates and returns the `MetaData` of a data set by parsing the features in an ARFF file to retrieve information
    about the features and outputs contained in a data set.

    :param features:    A list that contains a description of each feature in an ARFF file (including the outputs)
    :param outputs:     A list that contains the all outputs
    :return:            The `MetaData` that has been created
    """
    output_names = {output.name for output in outputs}
    outputs_at_start = False
    feature_list = []

    for feature in features:
        feature_name = __parse_feature_or_output_name(feature[0])

        if feature_name not in output_names:
            type_definition = feature[1]

            if isinstance(type_definition, list):
                feature_type = FeatureType.NOMINAL
                nominal_values = type_definition
            else:
                type_definition = str(type_definition).lower()
                nominal_values = None

                if type_definition == 'integer':
                    feature_type = FeatureType.ORDINAL
                elif type_definition in ('real', 'numeric'):
                    feature_type = FeatureType.NUMERICAL
                else:
                    raise ValueError('Encountered unsupported feature type: ' + type_definition)

            feature_list.append(Feature(feature_name, feature_type, nominal_values))
        elif len(feature_list) == 0:
            outputs_at_start = True

    meta_data = MetaData(feature_list, outputs, outputs_at_start)
    return meta_data


def __parse_feature_or_output_name(name: str) -> str:
    """
    Parses the name of an feature or output and removes forbidden characters.

    :param name:    The name of the feature or output
    :return:        The parsed name
    """
    name = name.strip()
    if name.startswith('\'') or name.startswith('\"'):
        name = name[1:]
    if name.endswith('\'') or name.endswith('\"'):
        name = name[:(len(name) - 1)]
    return name.replace('\\\'', '\'').replace('\\\"', '\"')


def __write_meta_data(xml_file, meta_data: MetaData):
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
