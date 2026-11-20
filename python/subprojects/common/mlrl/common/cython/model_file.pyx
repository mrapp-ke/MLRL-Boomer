"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to write models and metadata to model files and read them from such files. Model files are
HDF5 files that are kept in memory. Their content can be obtained as, or provided as, a `bytes` object.
"""
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport int64_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cython.operator cimport dereference

from mlrl.common.cython._types cimport float64, uint8
from mlrl.common.cython.output_space_info cimport IOutputSpaceInfo, OutputSpaceInfo, create_output_space_info
from mlrl.common.cython.probability_calibration cimport IJointProbabilityCalibrationModel, \
    IMarginalProbabilityCalibrationModel, JointProbabilityCalibrationModel, MarginalProbabilityCalibrationModel, \
    create_joint_probability_calibration_model, create_marginal_probability_calibration_model
from mlrl.common.cython.rule_model cimport IRuleModel, RuleModel, create_rule_model

from numbers import Integral, Real
from typing import Any

import numpy as np


cdef inline string _encode(object value):
    if not isinstance(value, str):
        raise TypeError('Expected a value of type str, but got a value of type ' + type(value).__name__)

    return value.encode('utf-8')


cdef inline str _decode(const string& value):
    return value.decode('utf-8')


def get_format_version() -> int:
    """
    Returns the version of the file format that is used for writing model files.

    :return: The version of the file format
    """
    return MODEL_FILE_FORMAT_VERSION


cdef class ModelFileWriter:
    """
    Allows to write models and metadata to a new model file.

    Groups are specified via paths relative to the root of the file, using "/" as a separator. An empty string refers to
    the root of the file. Groups are created automatically when they are written to for the first time.
    """

    def __cinit__(self):
        self.writer_ptr = createModelFileWriter()

    def set_attribute(self, group: str, name: str, value: Any):
        """
        Sets an attribute of a group. An existing attribute with the same name is replaced. Values of type `str` are
        stored as strings, values of type `bool` and other integers as integers, and other real numbers as floating
        point values.

        :param group:   The path of the group
        :param name:    The name of the attribute
        :param value:   The value of the attribute
        """
        if isinstance(value, str):
            self.writer_ptr.get().setStringAttribute(_encode(group), _encode(name), _encode(value))
        elif isinstance(value, (Integral, np.bool_)):
            self.writer_ptr.get().setIntAttribute(_encode(group), _encode(name), int(value))
        elif isinstance(value, Real):
            self.writer_ptr.get().setFloatAttribute(_encode(group), _encode(name), float(value))
        else:
            raise TypeError('Unable to write attribute "' + name + '" of unsupported type ' + type(value).__name__)

    def write_array(self, group: str, name: str, values: Any):
        """
        Writes a one-dimensional array to a group. Arrays with integer or boolean elements are stored as 64-bit
        integers, arrays with floating point elements as 64-bit floating point values, and arrays with string elements
        as strings.

        :param group:   The path of the group
        :param name:    The name of the array
        :param values:  A one-dimensional `np.ndarray` or a sequence that stores the elements of the array
        """
        cdef const int64_t[::1] int_values
        cdef const float64[::1] float_values
        cdef vector[string] string_values
        array = np.asarray(values)

        if array.ndim != 1:
            raise ValueError('Unable to write array "' + name + '": Expected a one-dimensional array, but got an array '
                             + 'with ' + str(array.ndim) + ' dimensions')

        if array.dtype.kind in ('U', 'S', 'O'):
            for value in array:
                if not isinstance(value, str):
                    raise TypeError('Unable to write array "' + name + '": Expected all elements to be strings, but '
                                    + 'got an element of type ' + type(value).__name__)
                string_values.push_back(_encode(value))

            self.writer_ptr.get().writeStringArray(_encode(group), _encode(name), string_values)
        elif array.dtype.kind in ('b', 'i', 'u'):
            if array.dtype.kind == 'u' and array.size > 0 and array.max() > np.iinfo(np.int64).max:
                raise OverflowError('Unable to write array "' + name + '": Values exceed the range of 64-bit integers')

            int_values = np.ascontiguousarray(array, dtype=np.int64)
            self.writer_ptr.get().writeInt64Array(_encode(group), _encode(name),
                                                  &int_values[0] if int_values.shape[0] > 0 else NULL,
                                                  int_values.shape[0])
        elif array.dtype.kind == 'f':
            float_values = np.ascontiguousarray(array, dtype=np.float64)
            self.writer_ptr.get().writeFloat64Array(_encode(group), _encode(name),
                                                    &float_values[0] if float_values.shape[0] > 0 else NULL,
                                                    float_values.shape[0])
        else:
            raise TypeError('Unable to write array "' + name + '" with elements of unsupported type '
                            + str(array.dtype))

    def write_rule_model(self, RuleModel rule_model not None):
        """
        Writes a rule-based model.

        :param rule_model: The `RuleModel` to be written
        """
        self.writer_ptr.get().writeRuleModel(dereference(rule_model.get_rule_model_ptr()))

    def write_output_space_info(self, OutputSpaceInfo output_space_info not None):
        """
        Writes information about the output space.

        :param output_space_info: The `OutputSpaceInfo` to be written
        """
        self.writer_ptr.get().writeOutputSpaceInfo(dereference(output_space_info.get_output_space_info_ptr()))

    def write_marginal_probability_calibration_model(self, MarginalProbabilityCalibrationModel model not None):
        """
        Writes a model for the calibration of marginal probabilities.

        :param model: The `MarginalProbabilityCalibrationModel` to be written
        """
        self.writer_ptr.get().writeMarginalProbabilityCalibrationModel(
            dereference(model.get_marginal_probability_calibration_model_ptr()))

    def write_joint_probability_calibration_model(self, JointProbabilityCalibrationModel model not None):
        """
        Writes a model for the calibration of joint probabilities.

        :param model: The `JointProbabilityCalibrationModel` to be written
        """
        self.writer_ptr.get().writeJointProbabilityCalibrationModel(
            dereference(model.get_joint_probability_calibration_model_ptr()))

    def finish(self) -> bytes:
        """
        Finishes writing the model file and returns its content. Afterwards, the writer cannot be used anymore.

        :return: A `bytes` object that stores the content of the model file
        """
        cdef vector[uint8] image = self.writer_ptr.get().finish()
        return PyBytes_FromStringAndSize(<char*>image.data(), image.size())


cdef class ModelFileReader:
    """
    Allows to read models and metadata from an existing model file.

    Groups are specified via paths relative to the root of the file, using "/" as a separator. An empty string refers to
    the root of the file.
    """

    def __cinit__(self, const uint8[::1] data not None):
        """
        :param data: A `bytes` object that stores the content of the model file. It is copied, i.e., it does not need
                     to outlive the reader
        """
        self.reader_ptr = createModelFileReader(&data[0] if data.shape[0] > 0 else NULL, data.shape[0])

    def get_format_version(self) -> int:
        """
        Returns the version of the file format that has been used for writing the model file.

        :return: The version of the file format
        """
        return self.reader_ptr.get().getFormatVersion()

    def get_library_version(self) -> str:
        """
        Returns the version of the library that has been used for writing the model file.

        :return: The version of the library
        """
        return _decode(self.reader_ptr.get().getLibraryVersion())

    def has_group(self, group: str) -> bool:
        """
        Returns whether a specific group exists or not.

        :param group:   The path of the group
        :return:        True, if the group exists, False otherwise
        """
        return self.reader_ptr.get().hasGroup(_encode(group))

    def has_dataset(self, group: str, name: str) -> bool:
        """
        Returns whether a group contains an array with a specific name or not.

        :param group:   The path of the group
        :param name:    The name of the array
        :return:        True, if the array exists, False otherwise
        """
        return self.reader_ptr.get().hasDataset(_encode(group), _encode(name))

    def has_attribute(self, group: str, name: str) -> bool:
        """
        Returns whether a group has an attribute with a specific name or not.

        :param group:   The path of the group
        :param name:    The name of the attribute
        :return:        True, if the attribute exists, False otherwise
        """
        return self.reader_ptr.get().hasAttribute(_encode(group), _encode(name))

    def get_attribute(self, group: str, name: str) -> int | float | str:
        """
        Returns the value of an attribute of a group.

        :param group:   The path of the group
        :param name:    The name of the attribute
        :return:        The value of the attribute
        """
        cdef string encoded_group = _encode(group)
        cdef string encoded_name = _encode(name)
        cdef ModelFileValueType value_type = self.reader_ptr.get().getAttributeType(encoded_group, encoded_name)

        if value_type == ModelFileValueType.INT:
            return self.reader_ptr.get().getIntAttribute(encoded_group, encoded_name)
        elif value_type == ModelFileValueType.FLOAT:
            return self.reader_ptr.get().getFloatAttribute(encoded_group, encoded_name)

        return _decode(self.reader_ptr.get().getStringAttribute(encoded_group, encoded_name))

    def get_attributes(self, group: str) -> dict[str, int | float | str]:
        """
        Returns the names and values of all attributes of a group.

        :param group:   The path of the group
        :return:        A dictionary that stores the names and values of the attributes, sorted by their names
        """
        cdef vector[string] names = self.reader_ptr.get().getAttributeNames(_encode(group))
        return {_decode(name): self.get_attribute(group, _decode(name)) for name in names}

    def read_array(self, group: str, name: str) -> np.ndarray:
        """
        Reads a one-dimensional array from a group.

        :param group:   The path of the group
        :param name:    The name of the array
        :return:        An `np.ndarray` of type `int64`, `float64` or `str` that stores the elements of the array
        """
        cdef string encoded_group = _encode(group)
        cdef string encoded_name = _encode(name)
        cdef ModelFileValueType value_type = self.reader_ptr.get().getArrayType(encoded_group, encoded_name)
        cdef vector[int64_t] int_values
        cdef vector[float64] float_values
        cdef vector[string] string_values

        if value_type == ModelFileValueType.INT:
            int_values = self.reader_ptr.get().readInt64Array(encoded_group, encoded_name)
            return np.asarray(<int64_t[:int_values.size()]>int_values.data() if int_values.size() > 0 else [],
                              dtype=np.int64).copy()
        elif value_type == ModelFileValueType.FLOAT:
            float_values = self.reader_ptr.get().readFloat64Array(encoded_group, encoded_name)
            return np.asarray(<float64[:float_values.size()]>float_values.data() if float_values.size() > 0 else [],
                              dtype=np.float64).copy()

        string_values = self.reader_ptr.get().readStringArray(encoded_group, encoded_name)
        return np.asarray([_decode(value) for value in string_values], dtype=str)

    def has_rule_model(self) -> bool:
        """
        Returns whether the model file contains a rule-based model or not.

        :return: True, if the model file contains a rule-based model, False otherwise
        """
        return self.reader_ptr.get().hasRuleModel()

    def read_rule_model(self) -> RuleModel:
        """
        Reads a rule-based model.

        :return: The `RuleModel` that has been read
        """
        cdef unique_ptr[IRuleModel] rule_model_ptr = self.reader_ptr.get().readRuleModel()
        return create_rule_model(move(rule_model_ptr))

    def has_output_space_info(self) -> bool:
        """
        Returns whether the model file contains information about the output space or not.

        :return: True, if the model file contains such information, False otherwise
        """
        return self.reader_ptr.get().hasOutputSpaceInfo()

    def read_output_space_info(self) -> OutputSpaceInfo:
        """
        Reads information about the output space.

        :return: The `OutputSpaceInfo` that has been read
        """
        cdef unique_ptr[IOutputSpaceInfo] output_space_info_ptr = self.reader_ptr.get().readOutputSpaceInfo()
        return create_output_space_info(move(output_space_info_ptr))

    def has_marginal_probability_calibration_model(self) -> bool:
        """
        Returns whether the model file contains a model for the calibration of marginal probabilities or not.

        :return: True, if the model file contains such a model, False otherwise
        """
        return self.reader_ptr.get().hasMarginalProbabilityCalibrationModel()

    def read_marginal_probability_calibration_model(self) -> MarginalProbabilityCalibrationModel:
        """
        Reads a model for the calibration of marginal probabilities.

        :return: The `MarginalProbabilityCalibrationModel` that has been read
        """
        cdef unique_ptr[IMarginalProbabilityCalibrationModel] model_ptr = (
            self.reader_ptr.get().readMarginalProbabilityCalibrationModel())
        return create_marginal_probability_calibration_model(move(model_ptr))

    def has_joint_probability_calibration_model(self) -> bool:
        """
        Returns whether the model file contains a model for the calibration of joint probabilities or not.

        :return: True, if the model file contains such a model, False otherwise
        """
        return self.reader_ptr.get().hasJointProbabilityCalibrationModel()

    def read_joint_probability_calibration_model(self) -> JointProbabilityCalibrationModel:
        """
        Reads a model for the calibration of joint probabilities.

        :return: The `JointProbabilityCalibrationModel` that has been read
        """
        cdef unique_ptr[IJointProbabilityCalibrationModel] model_ptr = (
            self.reader_ptr.get().readJointProbabilityCalibrationModel())
        return create_joint_probability_calibration_model(move(model_ptr))
