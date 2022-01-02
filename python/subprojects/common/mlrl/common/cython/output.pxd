from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython._data cimport CContiguousView
from mlrl.common.cython.input cimport CContiguousFeatureMatrixImpl, CsrFeatureMatrixImpl, LabelVector

from libcpp.memory cimport unique_ptr
from libcpp.forward_list cimport forward_list


cdef extern from "common/output/label_space_info.hpp" nogil:

    cdef cppclass ILabelSpaceInfo:
        pass


cdef extern from "common/output/label_space_info_no.hpp" nogil:

    cdef cppclass INoLabelSpaceInfo(ILabelSpaceInfo):
        pass


    unique_ptr[INoLabelSpaceInfo] createNoLabelSpaceInfo()


ctypedef void (*LabelVectorVisitor)(const LabelVector&)


cdef extern from "common/output/label_vector_set.hpp" nogil:

    cdef cppclass ILabelVectorSet(ILabelSpaceInfo):

        # Functions:

        void addLabelVector(unique_ptr[LabelVector] labelVectorPtr)

        void visit(LabelVectorVisitor)


    unique_ptr[ILabelVectorSet] createLabelVectorSet()


    cdef cppclass LabelVectorSetImpl"LabelVectorSet"(ILabelVectorSet):
        pass


cdef extern from *:
    """
    #include "common/output/label_vector_set.hpp"


    typedef void (*LabelVectorCythonVisitor)(void*, const LabelVector&);

    static inline LabelVectorSet::LabelVectorVisitor wrapLabelVectorVisitor(
            void* self, LabelVectorCythonVisitor visitor) {
        return [=](const LabelVector& labelVector) {
            visitor(self, labelVector);
        };
    }
    """

    ctypedef void (*LabelVectorCythonVisitor)(void*, const LabelVector&)

    LabelVectorVisitor wrapLabelVectorVisitor(void* self, LabelVectorCythonVisitor visitor)


cdef extern from "common/output/prediction_matrix_sparse_binary.hpp" nogil:

    cdef cppclass BinarySparsePredictionMatrix:

        ctypedef forward_list[uint32].const_iterator const_iterator

        # Functions:

        const_iterator row_cbegin(uint32 row)

        const_iterator row_cend(uint32 row)

        uint32 getNumRows()

        uint32 getNumCols()

        uint32 getNumNonZeroElements()


cdef extern from "common/output/predictor.hpp" nogil:

    cdef cppclass IPredictor[T]:

        # Functions:

        void predict(const CContiguousFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const LabelVectorSetImpl* labelVectors)

        void predict(const CsrFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const LabelVectorSetImpl* labelVectors)


cdef extern from "common/output/predictor_sparse.hpp" nogil:

    cdef cppclass ISparsePredictor[T](IPredictor[T]):

        # Functions:

        unique_ptr[BinarySparsePredictionMatrix] predictSparse(const CContiguousFeatureMatrixImpl& featureMatrix,
                                                               uint32 numLabels, const LabelVectorSetImpl* labelVectors)

        unique_ptr[BinarySparsePredictionMatrix] predictSparse(const CsrFeatureMatrixImpl& featureMatrix,
                                                               uint32 numLabels, const LabelVectorSetImpl* labelVectors)


cdef extern from "common/output/predictor_classification.hpp" nogil:

    cdef cppclass IClassificationPredictor(ISparsePredictor[uint8]):
        pass


    cdef cppclass IClassificationPredictorFactory:
        pass


cdef extern from "common/output/predictor_regression.hpp" nogil:

    cdef cppclass IRegressionPredictor(IPredictor[float64]):
        pass


    cdef cppclass IRegressionPredictorFactory:
        pass


cdef extern from "common/output/predictor_probability.hpp" nogil:

    cdef cppclass IProbabilityPredictor(IPredictor[float64]):
        pass


    cdef cppclass IProbabilityPredictorFactory:
        pass


cdef class LabelSpaceInfo:

    # Functions:

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self)


cdef class NoLabelSpaceInfo(LabelSpaceInfo):

    # Attributes:

    cdef unique_ptr[INoLabelSpaceInfo] label_space_info_ptr


cdef class LabelVectorSet(LabelSpaceInfo):

    # Attributes:

    cdef unique_ptr[ILabelVectorSet] label_vector_set_ptr


cdef class LabelVectorSetSerializer:

    # Attributes:

    cdef list state

    # Functions:

    cdef __visit_label_vector(self, const LabelVector& label_vector)


cdef class ClassificationPredictorFactory:

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IClassificationPredictorFactory] predictor_factory_ptr


cdef class RegressionPredictorFactory:

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IRegressionPredictorFactory] predictor_factory_ptr


cdef class ProbabilityPredictorFactory:

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IProbabilityPredictorFactory] predictor_factory_ptr


cdef class Predictor:
    pass


cdef class SparsePredictor(Predictor):
    pass


cdef class NumericalPredictor(Predictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IPredictor[float64]] predictor_ptr


cdef class BinaryPredictor(SparsePredictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[ISparsePredictor[uint8]] predictor_ptr
