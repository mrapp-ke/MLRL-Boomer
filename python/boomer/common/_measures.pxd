"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""


cdef extern from "cpp/measures/measure.hpp" nogil:

    cdef cppclass IMeasure:
        pass
