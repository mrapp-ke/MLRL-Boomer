"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides commonly used utility functions and structs.
"""
from boomer.algorithm._arrays cimport intp


cdef inline intp get_index(intp i, intp[::1] indices):
    """
    Retrieves and returns the i-th index from an array of indices, if such an array is available. Otherwise i is
    returned.

    :param i:       The position of the index that should be retrieved
    :param indices: An array of the dtype int, shape `(num_indices)`, representing the indices, or None
    :return:        A scalar of dtype int, representing the i-th index in the given array or i, if the array is None
    """
    if indices is None:
        return i
    else:
        return indices[i]
