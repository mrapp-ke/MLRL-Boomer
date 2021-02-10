"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""


cdef class Measure:
    """
    A wrapper for the pure virtual C++ class `IMeasure`.
    """

    cdef shared_ptr[IMeasure] get_measure_ptr(self):
        pass
