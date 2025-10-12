"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing datasets to ARFF files.
"""
import xml.etree.ElementTree as XmlTree

from pathlib import Path
from typing import Any, List
from xml.dom import minidom

import arff

from scipy.sparse import dok_array

from mlrl.testbed_sklearn.experiments.dataset import Attribute, AttributeType, TabularDataset

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.output.sinks.sink import DatasetFileSink
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import to_int_or_float
from mlrl.testbed.util.io import ENCODING_UTF8, open_writable_file

from mlrl.util.options import Options


class ArffFileSink(DatasetFileSink):
    """
    Allows to write a dataset to an ARFF file.
    """

    SUFFIX_ARFF = 'arff'

    SUFFIX_XML = 'xml'

    @staticmethod
    def __create_arff_data(dataset: TabularDataset) -> List[Any]:
        num_examples = dataset.num_examples

        if dataset.has_sparse_features and dataset.has_sparse_outputs:
            return [{} for _ in range(num_examples)]

        num_attributes = dataset.num_features + dataset.num_outputs
        attributes = dataset.features + dataset.outputs
        return [[ArffFileSink.__format_value(attributes[i], 0) for i in range(num_attributes)]
                for _ in range(num_examples)]

    @staticmethod
    def __format_value(attribute: Attribute, value: Any) -> Any:
        nominal = attribute.attribute_type != AttributeType.NUMERICAL and attribute.nominal_values
        nominal_values = attribute.nominal_values if nominal else None
        return nominal_values[int(value)] if nominal_values else to_int_or_float(value)

    @staticmethod
    def __fill_arff_data(dataset: TabularDataset, data: List[Any]):
        features = dataset.features

        for (example_index, feature_index), value in dok_array(dataset.x).items():
            feature = features[feature_index]
            data[example_index][feature_index] = ArffFileSink.__format_value(feature, value)

        num_features = dataset.num_features
        outputs = dataset.outputs

        for (example_index, output_index), value in dok_array(dataset.y).items():
            output = outputs[output_index]
            data[example_index][num_features + output_index] = ArffFileSink.__format_value(output, value)

    @staticmethod
    def __write_arff_file(file_path: Path, dataset: TabularDataset):
        features = dataset.features
        x_features = [(features[i].name, 'NUMERIC' if features[i].attribute_type == AttributeType.NUMERICAL
                       or features[i].nominal_values is None else features[i].nominal_values)
                      for i in range(dataset.num_features)]

        outputs = dataset.outputs
        y_features = [(outputs[i].name, 'NUMERIC' if outputs[i].attribute_type == AttributeType.NUMERICAL
                       or outputs[i].nominal_values is None else outputs[i].nominal_values)
                      for i in range(dataset.num_outputs)]

        data = ArffFileSink.__create_arff_data(dataset)
        ArffFileSink.__fill_arff_data(dataset, data)

        with open_writable_file(file_path) as arff_file:
            arff_file.write(
                arff.dumps({
                    'description': 'traindata',
                    'relation': 'traindata: -C ' + str(-dataset.num_outputs),
                    'attributes': x_features + y_features,
                    'data': data
                }))

    @staticmethod
    def __write_xml_file(file_path: Path, dataset: TabularDataset):
        root_element = XmlTree.Element('labels')
        root_element.set('xmlns', 'http://mulan.sourceforge.net/labels')

        for output in dataset.outputs:
            label_element = XmlTree.SubElement(root_element, 'label')
            label_element.set('name', output.name)

        with open_writable_file(file_path) as xml_file:
            xml_string = minidom.parseString(XmlTree.tostring(root_element)).toprettyxml(encoding=ENCODING_UTF8)
            xml_file.write(xml_string.decode(ENCODING_UTF8))

    def __init__(self, directory: Path, options: Options = Options(), create_directory: bool = False):
        """
        :param directory:           The path to the directory, where the ARFF file should be located
        :param options:             Options to be taken into account
        :param create_directory:    True, if the given directory should be created, if it does not exist, False
                                    otherwise
        """
        super().__init__(directory=directory,
                         suffix=self.SUFFIX_ARFF,
                         options=options,
                         create_directory=create_directory)

    # pylint: disable=unused-argument
    def _write_dataset_to_file(self, file_path: Path, state: ExperimentState, dataset: Dataset, **_):
        self.__write_arff_file(file_path=file_path, dataset=dataset)
        self.__write_xml_file(file_path=file_path.with_suffix('.' + self.SUFFIX_XML), dataset=dataset)
