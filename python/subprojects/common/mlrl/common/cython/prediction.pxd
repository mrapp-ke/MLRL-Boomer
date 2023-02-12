from mlrl.common.cython._types cimport uint8, uint32, float64

from libcpp.memory cimport unique_ptr
from libcpp cimport bool


cdef extern from "common/prediction/prediction_matrix_dense.hpp" nogil:

    cdef cppclass DensePredictionMatrix[T]:

        # Functions:

        uint32 getNumRows() const

        uint32 getNumCols() const

        T* release()


cdef extern from "common/prediction/prediction_matrix_sparse_binary.hpp" nogil:

    cdef cppclass BinarySparsePredictionMatrix:

        # Functions:

        uint32 getNumRows() const

        uint32 getNumCols() const

        uint32 getNumNonZeroElements() const

        uint32* releaseRowIndices()

        uint32* releaseColIndices()



cdef extern from "common/prediction/predictor.hpp" nogil:

    cdef cppclass IIncrementalPredictor[PredictionMatrix]:

        # Functions:

        uint32 getNumNext() const

        const PredictionMatrix& applyNext(uint32 stepSize)


    cdef cppclass IPredictor[PredictionMatrix]:

        # Functions:

        unique_ptr[PredictionMatrix] predict() const

        bool canPredictIncrementally() const

        unique_ptr[IIncrementalPredictor[PredictionMatrix]] createIncrementalPredictor() const


cdef extern from "common/prediction/predictor_binary.hpp" nogil:

    cdef cppclass IBinaryPredictor(IPredictor[DensePredictionMatrix[uint8]]):
        pass


    cdef cppclass ISparseBinaryPredictor(IPredictor[BinarySparsePredictionMatrix]):
        pass


cdef extern from "common/prediction/predictor_score.hpp" nogil:

    cdef cppclass IScorePredictor(IPredictor[DensePredictionMatrix[float64]]):
        pass


cdef extern from "common/prediction/predictor_probability.hpp" nogil:

    cdef cppclass IProbabilityPredictor(IPredictor[DensePredictionMatrix[float64]]):
        pass


cdef class BinaryPredictor:

    # Attributes:

    cdef unique_ptr[IBinaryPredictor] predictor_ptr


cdef class SparseBinaryPredictor:

    # Attributes:

    cdef unique_ptr[ISparseBinaryPredictor] predictor_ptr


cdef class ScorePredictor:

    # Attributes:

    cdef unique_ptr[IScorePredictor] predictor_ptr


cdef class ProbabilityPredictor:

    # Attributes:

    cdef unique_ptr[IProbabilityPredictor] predictor_ptr
