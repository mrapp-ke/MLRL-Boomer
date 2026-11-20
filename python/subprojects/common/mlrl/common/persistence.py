"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for storing the state of learners in model files and restoring it from such files.
"""

from numbers import Integral, Real
from typing import Any

import numpy as np

from mlrl.common.cython.model_file import ModelFileReader, ModelFileWriter

GROUP_LEARNER = 'learner'

GROUP_PARAMETERS = GROUP_LEARNER + '/parameters'

GROUP_FIT_STATE = GROUP_LEARNER + '/fit_state'

ATTRIBUTE_CLASS = 'class'

ATTRIBUTE_PACKAGE = 'package'

ATTRIBUTE_PACKAGE_VERSION = 'package_version'

DATASET_BOOLEAN_PARAMETERS = 'boolean_parameters'

DATASET_NONE_PARAMETERS = 'none_parameters'

SUFFIX_DTYPE = '_dtype'

SUFFIX_NDIM = '_ndim'


def write_parameters(writer: ModelFileWriter, parameters: dict[str, Any]):
    """
    Writes the hyperparameters of a learner to a model file. Parameters with value `None` and boolean parameters are
    recorded explicitly, such that their values can be restored exactly.

    :param writer:      The `ModelFileWriter` to be used
    :param parameters:  A dictionary that stores the names and values of the hyperparameters
    """
    boolean_parameters = []
    none_parameters = []

    for name, value in sorted(parameters.items()):
        if value is None:
            none_parameters.append(name)
        elif isinstance(value, (bool, np.bool_)):
            boolean_parameters.append(name)
            writer.set_attribute(GROUP_PARAMETERS, name, int(value))
        elif isinstance(value, str):
            writer.set_attribute(GROUP_PARAMETERS, name, value)
        elif isinstance(value, Integral):
            writer.set_attribute(GROUP_PARAMETERS, name, int(value))
        elif isinstance(value, Real):
            writer.set_attribute(GROUP_PARAMETERS, name, float(value))
        else:
            raise TypeError(
                f'Unable to save the hyperparameter "{name}": Only values of type bool, int, float, str or None are '
                + f'supported, but got a value of type {type(value).__name__}'
            )

    writer.write_array(GROUP_LEARNER, DATASET_BOOLEAN_PARAMETERS, np.asarray(boolean_parameters, dtype=str))
    writer.write_array(GROUP_LEARNER, DATASET_NONE_PARAMETERS, np.asarray(none_parameters, dtype=str))


def read_parameters(reader: ModelFileReader) -> dict[str, Any]:
    """
    Reads the hyperparameters of a learner from a model file.

    :param reader:  The `ModelFileReader` to be used
    :return:        A dictionary that stores the names and values of the hyperparameters
    """
    parameters: dict[str, Any] = (
        dict(reader.get_attributes(GROUP_PARAMETERS)) if reader.has_group(GROUP_PARAMETERS) else {}
    )

    for name in _read_names(reader, DATASET_BOOLEAN_PARAMETERS):
        if name not in parameters:
            raise ValueError(f'The model file is invalid: The value of the boolean hyperparameter "{name}" is missing')

        parameters[name] = bool(parameters[name])

    for name in _read_names(reader, DATASET_NONE_PARAMETERS):
        parameters[name] = None

    return parameters


def _read_names(reader: ModelFileReader, name: str) -> list[str]:
    if reader.has_dataset(GROUP_LEARNER, name):
        return [str(value) for value in reader.read_array(GROUP_LEARNER, name)]

    return []


def write_numpy_value(writer: ModelFileWriter, group: str, name: str, value: Any):
    """
    Writes a scalar value or a one-dimensional array to a model file, such that it can be restored with its original
    NumPy data type. Only numerical, boolean and string values are supported.

    :param writer:  The `ModelFileWriter` to be used
    :param group:   The path of the group
    :param name:    The name of the value
    :param value:   The value to be written
    """
    array = np.asarray(value)

    if array.ndim > 1:
        raise TypeError(f'Unable to save "{name}": Only scalars and one-dimensional arrays are supported')

    try:
        writer.write_array(group, name, array.reshape(-1))
    except (TypeError, OverflowError) as error:
        raise TypeError(
            f'Unable to save "{name}": Only numerical, boolean and string values are supported, but got values of '
            + f'type {array.dtype}'
        ) from error

    writer.set_attribute(group, name + SUFFIX_DTYPE, array.dtype.str)
    writer.set_attribute(group, name + SUFFIX_NDIM, array.ndim)


def read_numpy_value(reader: ModelFileReader, group: str, name: str) -> Any:
    """
    Reads a scalar value or a one-dimensional array that has been written via the function `write_numpy_value`.

    :param reader:  The `ModelFileReader` to be used
    :param group:   The path of the group
    :param name:    The name of the value
    :return:        The value that has been read, i.e., a NumPy scalar or a one-dimensional `np.ndarray`
    """
    dtype = np.dtype(reader.get_attribute(group, name + SUFFIX_DTYPE))
    ndim = reader.get_attribute(group, name + SUFFIX_NDIM)
    array = reader.read_array(group, name).astype(dtype)

    if ndim == 0:
        if array.shape != (1,):
            raise ValueError(f'The model file is invalid: Expected "{name}" to be a scalar value')

        return array.reshape(())[()]

    return array


def has_numpy_value(reader: ModelFileReader, group: str, name: str) -> bool:
    """
    Returns whether a value that has been written via the function `write_numpy_value` exists or not.

    :param reader:  The `ModelFileReader` to be used
    :param group:   The path of the group
    :param name:    The name of the value
    :return:        True, if the value exists, False otherwise
    """
    return reader.has_dataset(group, name)
