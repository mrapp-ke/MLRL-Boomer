"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""


cdef extern from "common/measures/measure.hpp" nogil:

    cdef cppclass IMeasure:
        pass
