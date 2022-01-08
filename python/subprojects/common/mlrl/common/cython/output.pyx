"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._arrays cimport array_uint32, c_matrix_uint8, c_matrix_float64
from mlrl.common.cython.input cimport CContiguousFeatureMatrix, CContiguousFeatureMatrixImpl, CsrFeatureMatrixImpl, \
    CsrFeatureMatrix, RowWiseLabelMatrix, IRowWiseLabelMatrix
from mlrl.common.cython.model cimport RuleModel

from libcpp.memory cimport make_unique
from libcpp.utility cimport move

from cython.operator cimport dereference

from scipy.sparse import csr_matrix
import numpy as np

SERIALIZATION_VERSION = 1


cdef class LabelSpaceInfo:
    """
    A wrapper for the pure virtual C++ class `ILabelSpaceInfo`.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        pass


cdef class NoLabelSpaceInfo(LabelSpaceInfo):
    """
    A wrapper for the pure virtual C++ class `INoLabelSpaceInfo`.
    """

    def __cinit__(self):
        self.label_space_info_ptr = createNoLabelSpaceInfo()

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        return self.label_space_info_ptr.get()

    def __reduce__(self):
        return (NoLabelSpaceInfo, ())


cdef class LabelVectorSet(LabelSpaceInfo):
    """
    A wrapper for the pure virtual C++ class `ILabelVectorSet`.
    """

    def __cinit__(self):
        self.label_vector_set_ptr = createLabelVectorSet()

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        return self.label_vector_set_ptr.get()

    @classmethod
    def create(cls, RowWiseLabelMatrix label_matrix):
        cdef IRowWiseLabelMatrix* label_matrix_ptr = label_matrix.get_row_wise_label_matrix_ptr()
        cdef uint32 num_rows = label_matrix_ptr.getNumRows()
        cdef uint32 num_cols = label_matrix_ptr.getNumCols()
        cdef unique_ptr[ILabelVectorSet] label_vector_set_ptr = createLabelVectorSet()
        cdef unique_ptr[LabelVector] label_vector_ptr
        cdef uint32 i

        for i in range(num_rows):
            label_vector_ptr = label_matrix_ptr.createLabelVector(i)
            label_vector_set_ptr.get().addLabelVector(move(label_vector_ptr))

        cdef LabelVectorSet label_vector_set = LabelVectorSet.__new__(LabelVectorSet)
        label_vector_set.label_vector_set_ptr = move(label_vector_set_ptr)
        return label_vector_set

    def __reduce__(self):
        cdef LabelVectorSetSerializer serializer = LabelVectorSetSerializer.__new__(LabelVectorSetSerializer)
        cdef object state = serializer.serialize(self)
        return (LabelVectorSet, (), state)

    def __setstate__(self, state):
        cdef LabelVectorSetSerializer serializer = LabelVectorSetSerializer.__new__(LabelVectorSetSerializer)
        serializer.deserialize(self, state)


cdef inline unique_ptr[LabelVector] __create_label_vector(list state):
    cdef uint32 num_elements = len(state)
    cdef unique_ptr[LabelVector] label_vector_ptr = make_unique[LabelVector](num_elements)
    cdef LabelVector.iterator iterator = label_vector_ptr.get().begin()
    cdef uint32 i, label_index

    for i in range(num_elements):
        label_index = state[i]
        iterator[i] = label_index

    return move(label_vector_ptr)


cdef class LabelVectorSetSerializer:
    """
    Allows to serialize and deserialize the label vectors that are stored by a `LabelVectorSet`.
    """

    cdef __visit_label_vector(self, const LabelVector& label_vector):
        cdef list label_vector_state = []
        cdef uint32 num_elements = label_vector.getNumElements()
        cdef LabelVector.const_iterator iterator = label_vector.cbegin()
        cdef uint32 i, label_index

        for i in range(num_elements):
            label_index = iterator[i]
            label_vector_state.append(label_index)

        self.state.append(label_vector_state)

    def serialize(self, LabelVectorSet label_vector_set not None):
        """
        Creates and returns a state, which may be serialized using Python's pickle mechanism, from the label vectors
        that are stored by a given `LabelVectorSet`.

        :param label_vector_set:    The set that stores the label vectors to be serialized
        :return:                    The state that has been created
        """
        self.state = []
        cdef ILabelVectorSet* label_vector_set_ptr = label_vector_set.label_vector_set_ptr.get()
        label_vector_set_ptr.visit(wrapLabelVectorVisitor(<void*>self,
                                                          <LabelVectorCythonVisitor>self.__visit_label_vector))
        return (SERIALIZATION_VERSION, self.state)

    def deserialize(self, LabelVectorSet label_vector_set not None, object state not None):
        """
        Deserializes the label vectors that are stored by a given state and adds them to a `LabelVectorSet`.

        :param label_vector_set:    The set, the deserialized rules should be added to
        :param state:               A state that has previously been created via the function `serialize`
        """
        cdef int version = state[0]

        if version != SERIALIZATION_VERSION:
            raise AssertionError(
                'Version of the serialized LabelVectorSet is ' + str(version) + ', expected ' + str(SERIALIZATION_VERSION))

        cdef list label_vector_list = state[1]
        cdef uint32 num_label_vectors = len(label_vector_list)
        cdef ILabelVectorSet* label_vector_set_ptr = label_vector_set.label_vector_set_ptr.get()
        cdef list label_vector_state
        cdef uint32 i

        for i in range(num_label_vectors):
            label_vector_state = label_vector_list[i]
            label_vector_set_ptr.addLabelVector(move(__create_label_vector(label_vector_state)))


cdef class ClassificationPredictorFactory:
    """
    A wrapper for the pure virtual C++ class `IClassificationPredictorFactory`.
    """

    def create(self, RuleModel model not None, LabelSpaceInfo label_space_info not None) -> BinaryPredictor:
        cdef BinaryPredictor predictor = BinaryPredictor.__new__(BinaryPredictor)
        predictor.num_labels = self.num_labels
        predictor.predictor_ptr = <unique_ptr[ISparsePredictor[uint8]]>model.model_ptr.get().createClassificationPredictor(
            dereference(self.predictor_factory_ptr.get()), dereference(label_space_info.get_label_space_info_ptr()))
        return predictor


cdef class RegressionPredictorFactory:
    """
    A wrapper for the pure virtual C++ class `IRegressionPredictorFactory`.
    """

    def create(self, RuleModel model not None, LabelSpaceInfo label_space_info not None) -> NumericalPredictor:
        cdef NumericalPredictor predictor = NumericalPredictor.__new__(NumericalPredictor)
        predictor.num_labels = self.num_labels
        predictor.predictor_ptr = <unique_ptr[IPredictor[float64]]>model.model_ptr.get().createRegressionPredictor(
            dereference(self.predictor_factory_ptr.get()), dereference(label_space_info.get_label_space_info_ptr()))
        return predictor


cdef class ProbabilityPredictorFactory:
    """
    A wrapper for the pure virtual C++ class `IProbabilityPredictorFactory`.
    """

    def create(self, RuleModel model not None, LabelSpaceInfo label_space_info not None) -> NumericalPredictor:
        cdef NumericalPredictor predictor = NumericalPredictor.__new__(NumericalPredictor)
        predictor.num_labels = self.num_labels
        predictor.predictor_ptr = <unique_ptr[IPredictor[float64]]>model.model_ptr.get().createProbabilityPredictor(
            dereference(self.predictor_factory_ptr.get()), dereference(label_space_info.get_label_space_info_ptr()))
        return predictor


cdef class Predictor:
    """
    A wrapper for the pure virtual C++ class `IPredictor`.
    """

    def predict_dense(self, CContiguousFeatureMatrix feature_matrix not None) -> np.ndarray:
        """
        Obtains and returns dense predictions for given examples in a feature matrix that uses a C-contiguous array.

        :param feature_matrix:  A `CContiguousFeatureMatrix` that stores the examples to predict for
        :return:                A `np.ndarray`, shape `(num_examples, num_labels)`, that stores the predictions for
                                individual examples and labels
        """
        pass

    def predict_dense_csr(self, CsrFeatureMatrix feature_matrix not None) -> np.ndarray:
        """
        Obtains and returns dense predictions for given examples in a feature matrix that uses the compressed sparse row
        (CSR) format.

        :param feature_matrix:  A `CsrFeatureMatrix` that stores the examples to predict for
        :return:                A `np.ndarray`, shape `(num_examples, num_labels)`, that stores the predictions for
                                individual examples and labels
        """
        pass


cdef class SparsePredictor(Predictor):
    """
    A wrapper for the pure virtual C++ class `ISparsePredictor`.
    """

    def predict_sparse(self, CContiguousFeatureMatrix feature_matrix not None) -> csr_matrix:
        """
        Obtains and returns sparse predictions for given examples in a feature matrix that uses a C-contiguous array.

        :param feature_matrix:  A `CContiguousFeatureMatrix` that stores the examples to predict for
        :return:                A `scipy.sparse.csr_matrix`, shape `(num_examples, num_labels)`, that stores the
                                predictions for individual examples and labels
        """
        pass

    def predict_sparse_csr(self, CsrFeatureMatrix feature_matrix not None) -> csr_matrix:
        """
        Obtains and returns dense predictions for given examples in a feature matrix that uses the compressed sparse row
        (CSR) format.

        :param feature_matrix:  A `CsrFeatureMatrix` that stores the examples to predict for
        :return:                A `scipy.sparse.csr_matrix`, shape `(num_examples, num_labels)`, that stores the
                                predictions for individual examples and labels
        """
        pass


cdef class NumericalPredictor(Predictor):
    """
    A base class for all classes that allow to predict numerical scores for given query examples.
    """

    def predict(self, CContiguousFeatureMatrix feature_matrix not None):
        cdef CContiguousFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_examples = feature_matrix_ptr.getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef unique_ptr[DensePredictionMatrix[float64]] prediction_matrix_ptr = self.predictor_ptr.get().predict(
            dereference(feature_matrix_ptr), num_labels)
        cdef float64* array = prediction_matrix_ptr.get().release()
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(array, num_examples, num_labels)
        return np.asarray(prediction_matrix)

    def predict_csr(self, CsrFeatureMatrix feature_matrix not None):
        cdef CsrFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_examples = feature_matrix_ptr.getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef unique_ptr[DensePredictionMatrix[float64]] prediction_matrix_ptr = self.predictor_ptr.get().predict(
            dereference(feature_matrix_ptr), num_labels)
        cdef float64* array = prediction_matrix_ptr.get().release()
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(array, num_examples, num_labels)
        return np.asarray(prediction_matrix)


cdef inline object __create_csr_matrix(BinarySparsePredictionMatrix* prediction_matrix):
    cdef uint32 num_rows = prediction_matrix.getNumRows()
    cdef uint32 num_cols = prediction_matrix.getNumCols()
    cdef uint32 num_non_zero_elements = prediction_matrix.getNumNonZeroElements()
    cdef uint32* row_indices = prediction_matrix.releaseRowIndices()
    cdef uint32* col_indices = prediction_matrix.releaseColIndices()
    data = np.ones(shape=(num_non_zero_elements), dtype=np.uint8) if num_non_zero_elements > 0 else np.asarray([])
    indices = np.asarray(array_uint32(col_indices, num_non_zero_elements) if num_non_zero_elements > 0 else [])
    indptr = np.asarray(array_uint32(row_indices, num_rows + 1))
    return csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))


cdef class BinaryPredictor(SparsePredictor):
    """
    A base class for all classes that allow to predict binary values for given query examples.
    """

    def predict_dense(self, CContiguousFeatureMatrix feature_matrix not None):
        cdef CContiguousFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_examples = feature_matrix_ptr.getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef IPredictor[uint8]* predictor_ptr = self.predictor_ptr.get()
        cdef unique_ptr[DensePredictionMatrix[uint8]] prediction_matrix_ptr = predictor_ptr.predict(
            dereference(feature_matrix_ptr), num_labels)
        cdef uint8* array = prediction_matrix_ptr.get().release()
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(array, num_examples, num_labels)
        return np.asarray(prediction_matrix)

    def predict_dense_csr(self, CsrFeatureMatrix feature_matrix not None):
        cdef CsrFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_examples = feature_matrix_ptr.getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef IPredictor[uint8]* predictor_ptr = self.predictor_ptr.get()
        cdef unique_ptr[DensePredictionMatrix[uint8]] prediction_matrix_ptr = predictor_ptr.predict(
            dereference(feature_matrix_ptr), num_labels)
        cdef uint8* array = prediction_matrix_ptr.get().release()
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(array, num_examples, num_labels)
        return np.asarray(prediction_matrix)

    def predict_sparse(self, CContiguousFeatureMatrix feature_matrix not None) -> csr_matrix:
        cdef CContiguousFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_labels = self.num_labels
        cdef ISparsePredictor[uint8]* predictor_ptr = self.predictor_ptr.get()
        cdef unique_ptr[BinarySparsePredictionMatrix] prediction_matrix_ptr = predictor_ptr.predictSparse(
            dereference(feature_matrix_ptr), num_labels)
        return __create_csr_matrix(prediction_matrix_ptr.get())

    def predict_sparse_csr(self, CsrFeatureMatrix feature_matrix not None) -> csr_matrix:
        cdef CsrFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_labels = self.num_labels
        cdef ISparsePredictor[uint8]* predictor_ptr = self.predictor_ptr.get()
        cdef unique_ptr[BinarySparsePredictionMatrix] prediction_matrix_ptr = predictor_ptr.predictSparse(
            dereference(feature_matrix_ptr), num_labels)
        return __create_csr_matrix(prediction_matrix_ptr.get())
