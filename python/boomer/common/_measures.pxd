"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""


cdef extern from "cpp/measures/measure.h" nogil:

    cdef cppclass IMeasure:
        pass
