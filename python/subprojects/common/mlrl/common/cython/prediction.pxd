from mlrl.common.cython._types cimport uint8, uint32, float64

from libcpp.memory cimport unique_ptr


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



cdef extern from "common/prediction/predictor.hpp" nogil:

    cdef cppclass IPredictor[PredictionMatrix]:

        # Functions:

        unique_ptr[PredictionMatrix] predict() const


cdef extern from "common/prediction/predictor_label.hpp" nogil:

    cdef cppclass ILabelPredictor(IPredictor[DensePredictionMatrix[uint8]]):
        pass


    cdef cppclass ISparseLabelPredictor(IPredictor[BinarySparsePredictionMatrix]):
        pass


cdef extern from "common/prediction/predictor_score.hpp" nogil:

    cdef cppclass IScorePredictor(IPredictor[DensePredictionMatrix[float64]]):
        pass


cdef extern from "common/prediction/predictor_probability.hpp" nogil:

    cdef cppclass IProbabilityPredictor(IPredictor[DensePredictionMatrix[uint8]]):
        pass
