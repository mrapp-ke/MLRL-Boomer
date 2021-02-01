"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""


cdef extern from "cpp/indices/index_vector.h" nogil:

    cdef cppclass IIndexVector:
        pass
