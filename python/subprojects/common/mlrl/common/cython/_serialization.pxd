"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for pickling native objects as HDF5 images.
"""
from cpython.bytes cimport PyBytes_FromStringAndSize
from libcpp.vector cimport vector

from mlrl.common.cython._types cimport uint8


cdef inline bytes image_to_bytes(const vector[uint8]& image):
    """
    Converts the image of a model file into a `bytes` object.

    :param image:   A reference to a `vector` that stores the image of a model file
    :return:        A `bytes` object that stores the image
    """
    return PyBytes_FromStringAndSize(<const char*>image.data(), image.size())


cdef inline void check_pickled_state(object state, str class_name) except *:
    """
    Raises a `ValueError` if the state of a pickled object has not been created by the current version of the library,
    i.e., if it is not a `bytes` object that stores the image of a model file.

    :param state:       The state of the pickled object
    :param class_name:  The name of the class of the pickled object
    """
    if not isinstance(state, bytes):
        raise ValueError('Unable to unpickle an object of type ' + class_name + ', because it has been pickled by an '
                         + 'older version of the library. Pickled models are not compatible across these versions. '
                         + 'Please re-train the model.')
