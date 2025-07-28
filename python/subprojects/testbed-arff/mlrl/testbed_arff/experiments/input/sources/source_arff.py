"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading datasets from ARFF files.
"""
import logging as log

from functools import cached_property
from pathlib import Path
from typing import Any, List, Optional, Set
from xml.dom import minidom

import arff
import numpy as np

from scipy.sparse import coo_array, csc_array, sparray

from mlrl.testbed_arff.experiments.output.sinks.sink_arff import ArffFileSink

from mlrl.testbed_sklearn.experiments.dataset import Attribute, AttributeType, TabularDataset

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.input.data import DatasetInputData
from mlrl.testbed.experiments.input.sources.source import DatasetFileSource
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.io import open_readable_file


def normalize_attribute_name(name: str) -> str:
    """
    Normalizes the name of an attribute by removing forbidden characters.

    :param name:    The name to be normalized
    :return:        The normalized name
    """
    name = name.strip()
    if name.startswith('\'') or name.startswith('"'):
        name = name[1:]
    if name.endswith('\'') or name.endswith('"'):
        name = name[:(len(name) - 1)]
    return name.replace('\\\'', '\'').replace('\\"', '"')


class ArffFileSource(DatasetFileSource):
    """
    Allows to read a dataset from an ARFF file.
    """

    class ArffFile:
        """
        Provides access to the content of an ARFF file.
        """

        def __init__(self, matrix: sparray, arff_attributes: List[Any], relation: str):
            """
            param matrix:           The data matrix that is stored in the file
            param arff_attributes:  The attributes defined in the file
            param relation:         The @relation declaration contained in the file
            """
            self.matrix = matrix
            self.arff_attributes = arff_attributes
            self.relation = relation

        @staticmethod
        def from_file(file_path: Path, sparse: bool, dtype: np.dtype) -> 'ArffFileSource.ArffFile':
            """
            Loads the content of an ARFF file.

            :param file_path:   The path to the ARFF file
            :param sparse:      True, if the ARFF file is given in sparse format, False otherwise. If the given format
                                is incorrect, an `arff.BadLayout` is raised
            :param dtype:       The type of the data matrix to be read from the file
            :return:            A dictionary that stores the content of the ARFF file
            """
            with open_readable_file(file_path) as arff_file:
                arff_dict = arff.load(arff_file, encode_nominal=True, return_type=arff.COO if sparse else arff.DENSE)

                if sparse:
                    data = arff_dict['data']
                    values = data[0]
                    row_indices = data[1]
                    col_indices = data[2]
                    shape = (max(row_indices) + 1, max(col_indices) + 1)
                    matrix = coo_array((values, (row_indices, col_indices)), shape=shape, dtype=dtype)
                    matrix = matrix.tocsc()
                else:
                    data = arff_dict['data']
                    matrix = csc_array(data, dtype=dtype)

            attributes = arff_dict['attributes']
            relation = arff_dict['relation']
            return ArffFileSource.ArffFile(matrix, arff_attributes=attributes, relation=relation)

        @cached_property
        def attributes(self) -> List[Attribute]:
            """
            A list that contains all attributes defined in the ARFF file.
            """
            attributes = []

            for attribute in self.arff_attributes:
                attribute_name = normalize_attribute_name(attribute[0])
                type_definition = attribute[1]

                if isinstance(type_definition, list):
                    attribute_type = AttributeType.NOMINAL
                    nominal_values = type_definition
                else:
                    type_definition = str(type_definition).lower()
                    nominal_values = None

                    if type_definition == 'integer':
                        attribute_type = AttributeType.ORDINAL
                    elif type_definition in ('real', 'numeric'):
                        attribute_type = AttributeType.NUMERICAL
                    else:
                        raise ValueError('Encountered unsupported attribute type: ' + type_definition)

                attribute = Attribute(name=attribute_name, attribute_type=attribute_type, nominal_values=nominal_values)
                attributes.append(attribute)

            return attributes

    class ArffDataset:
        """
        Provides access to the content of an ARFF file and the corresponding Mulan XML file, if available.
        """

        def __parse_output_names_from_relation(self) -> Set[str]:
            parameter_name = '-C '
            arff_file = self.arff_file
            relation = arff_file.relation
            index = relation.index(parameter_name)
            parameter_value = relation[index + len(parameter_name):]
            index = parameter_value.find(' ')

            if index >= 0:
                parameter_value = parameter_value[:index]

            num_outputs = int(parameter_value)
            attributes = arff_file.attributes
            return {normalize_attribute_name(attributes[i].name) for i in range(num_outputs)}

        def __init__(self, arff_file: 'ArffFileSource.ArffFile', output_names: Optional[Set[str]]):
            """
            :param arff_file:       The content of the ARFF file
            :param output_names:    The names of all outputs contained in the dataset
            """
            self.arff_file = arff_file
            self.output_names = output_names if output_names else self.__parse_output_names_from_relation()

        @staticmethod
        def from_file(arff_file: 'ArffFileSource.ArffFile', file_path: Path) -> 'ArffFileSource.ArffDataset':
            """
            Creates and returns an ARFF dataset from given ARFF file and a corresponding Mulan XML file, if available.


            :param arff_file:       The content of the ARFF file
            :param file_path:       The path to the XML file
            :return:                The ARFF dataset that has been created
            """
            if file_path.is_file():
                log.debug('Parsing meta-data from file \"%s\"...', file_path)
                xml_doc = minidom.parse(str(file_path))
                tags = xml_doc.getElementsByTagName('label')
                output_names = {normalize_attribute_name(tag.getAttribute('name')) for tag in tags}
            else:
                output_names = None
                log.debug(
                    'Mulan XML file \"%s\" does not exist. If possible, information about the dataset\'s outputs is '
                    + 'parsed from the ARFF file\'s @relation declaration as intended by the MEKA dataset format...',
                    file_path)

            return ArffFileSource.ArffDataset(arff_file=arff_file, output_names=output_names)

        @cached_property
        def features(self) -> List[Attribute]:
            """
            A list that stores all features contained in the dataset.
            """
            return [attribute for attribute in self.arff_file.attributes if attribute.name not in self.output_names]

        @cached_property
        def outputs(self) -> List[Attribute]:
            """
            A list that stores all outputs contained in the dataset.
            """
            return [attribute for attribute in self.arff_file.attributes if attribute.name in self.output_names]

        @property
        def outputs_at_start(self) -> bool:
            """
            True, if the outputs are defined before the features, False otherwise.
            """
            attributes = self.arff_file.attributes
            return bool(attributes) and attributes[0].name in self.output_names

        @property
        def feature_matrix(self) -> sparray:
            """
            The feature matrix contained in the dataset.
            """
            num_outputs = len(self.outputs)
            matrix = self.arff_file.matrix
            return matrix[:, num_outputs:] if self.outputs_at_start else matrix[:, :-num_outputs]

        @property
        def output_matrix(self) -> sparray:
            """
            The output matrix contained in the dataset.
            """
            num_outputs = len(self.outputs)
            matrix = self.arff_file.matrix
            return matrix[:, :num_outputs] if self.outputs_at_start else matrix[:, -num_outputs:]

    @staticmethod
    def __read_arff_file(file_path: Path, dtype: np.dtype) -> ArffFile:
        try:
            return ArffFileSource.ArffFile.from_file(file_path, sparse=True, dtype=dtype)
        except arff.BadLayout:
            return ArffFileSource.ArffFile.from_file(file_path, sparse=False, dtype=dtype)

    def __init__(self, directory: Path):
        """
        :param directory: The path to the directory of the file
        """
        super().__init__(directory=directory, suffix=ArffFileSink.SUFFIX_ARFF)

    def _read_dataset_from_file(self, state: ExperimentState, file_path: Path,
                                input_data: DatasetInputData) -> Optional[Dataset]:
        properties = input_data.properties
        problem_domain = state.problem_domain
        arff_file = self.__read_arff_file(file_path=file_path, dtype=problem_domain.feature_dtype)
        xml_file_path = file_path.with_name(properties.file_name + '.' + ArffFileSink.SUFFIX_XML)
        arff_dataset = ArffFileSource.ArffDataset.from_file(arff_file=arff_file, file_path=xml_file_path)
        return TabularDataset(x=arff_dataset.feature_matrix.tolil(),
                              y=arff_dataset.output_matrix.astype(problem_domain.output_dtype).tolil(),
                              features=arff_dataset.features,
                              outputs=arff_dataset.outputs)
