from libcpp cimport bool

from mlrl.common.cython._types cimport uint32


cdef extern from "mlrl/common/input/output_matrix.hpp" nogil:

    cdef cppclass IOutputMatrix:

        # Functions:

        uint32 getNumExamples() const

        uint32 getNumOutputs() const

        bool isSparse() const


cdef class OutputMatrix:

    # Functions:

    cdef IOutputMatrix* get_output_matrix_ptr(self)
