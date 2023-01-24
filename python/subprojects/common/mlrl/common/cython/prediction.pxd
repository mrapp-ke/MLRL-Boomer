from mlrl.common.cython._types cimport uint8, uint32, float64


cdef extern from "common/prediction/prediction_matrix_dense.hpp" nogil:

    cdef cppclass DensePredictionMatrix[T]:

        # Functions:

        T* release()


cdef extern from "common/prediction/prediction_matrix_sparse_binary.hpp" nogil:

    cdef cppclass BinarySparsePredictionMatrix:

        # Functions:

        uint32 getNumNonZeroElements() const

        uint32* releaseRowIndices()

        uint32* releaseColIndices()
