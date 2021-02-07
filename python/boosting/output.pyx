"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from common._types cimport uint32
from common.input cimport CContiguousLabelMatrix, CContiguousLabelMatrixImpl, ILabelMatrix
from boosting.losses_label_wise cimport LabelWiseLoss
from boosting.losses_example_wise cimport ExampleWiseLoss

from cython.operator cimport dereference, postincrement

from libcpp.memory cimport shared_ptr, make_unique, dynamic_pointer_cast
from libcpp.utility cimport move

SERIALIZATION_VERSION = 1


cdef class LabelWiseClassificationPredictor(AbstractClassificationPredictor):
    """
    A wrapper for the C++ class `LabelWiseClassificationPredictor`.
    """

    def __cinit__(self, uint32 num_labels, float64 threshold):
        """
        :param num_labels:  The total number of available labels
        :param thresholds:  The threshold to be used for making predictions
        """
        self.num_labels = num_labels
        self.threshold = threshold
        self.predictor_ptr = <unique_ptr[IPredictor[uint8]]>make_unique[LabelWiseClassificationPredictorImpl](threshold)

    def __reduce__(self):
        return (LabelWiseClassificationPredictor, (self.num_labels, self.threshold))


cdef inline shared_ptr[IMeasure] __get_measure_ptr(object measure):
    cdef shared_ptr[IMeasure] measure_ptr
    cdef LabelWiseLoss label_wise_loss
    cdef ExampleWiseLoss example_wise_loss

    if isinstance(measure, LabelWiseLoss):
        label_wise_loss = measure
        measure_ptr = <shared_ptr[IMeasure]>label_wise_loss.loss_function_ptr
    elif isinstance(measure, ExampleWiseLoss):
        example_wise_loss = measure
        measure_ptr = <shared_ptr[IMeasure]>example_wise_loss.loss_function_ptr
    else:
        raise ValueError('Unknown type of measure: ' + type(measure).__name__)

    return measure_ptr


cdef class ExampleWiseClassificationPredictor(AbstractClassificationPredictor):
    """
    A wrapper for the C++ class `ExampleWiseClassificationPredictor`.
    """

    def __cinit__(self, uint32 num_labels, object measure):
        """
        :param num_labels:  The total number of available labels
        :param measure:     The measure to be used
        """
        self.num_labels = num_labels
        self.measure = measure

    @classmethod
    def create(cls, CContiguousLabelMatrix label_matrix, object measure):
        cdef shared_ptr[IMeasure] measure_ptr = __get_measure_ptr(measure)
        cdef shared_ptr[CContiguousLabelMatrixImpl] label_matrix_ptr = dynamic_pointer_cast[CContiguousLabelMatrixImpl, ILabelMatrix](
            label_matrix.label_matrix_ptr)
        cdef uint32 num_rows = label_matrix_ptr.get().getNumRows()
        cdef uint32 num_cols = label_matrix_ptr.get().getNumCols()
        cdef unique_ptr[ExampleWiseClassificationPredictorImpl] predictor_ptr = make_unique[ExampleWiseClassificationPredictorImpl](
            measure_ptr)
        cdef unique_ptr[LabelVector] label_vector_ptr
        cdef uint8 value
        cdef uint32 i, j

        for i in range(num_rows):
            label_vector_ptr = make_unique[LabelVector]()

            for j in range(num_cols):
                value = label_matrix_ptr.get().getValue(i, j)

                if value:
                    label_vector_ptr.get().setValue(j)

            predictor_ptr.get().addLabelVector(move(label_vector_ptr))

        cdef ExampleWiseClassificationPredictor predictor = ExampleWiseClassificationPredictor.__new__(
            ExampleWiseClassificationPredictor, num_cols, measure)
        predictor.predictor_ptr = <unique_ptr[IPredictor[uint8]]>move(predictor_ptr)
        return predictor

    @classmethod
    def create_lil(cls, uint32 num_labels, list[::1] rows, object measure):
        cdef shared_ptr[IMeasure] measure_ptr = __get_measure_ptr(measure)
        cdef uint32 num_rows = rows.shape[0]
        cdef unique_ptr[ExampleWiseClassificationPredictorImpl] predictor_ptr = make_unique[ExampleWiseClassificationPredictorImpl](
            measure_ptr)
        cdef unique_ptr[LabelVector] label_vector_ptr
        cdef list col_indices
        cdef uint32 i, j

        for i in range(num_rows):
            label_vector_ptr = make_unique[LabelVector]()
            col_indices = rows[i]

            for j in col_indices:
                label_vector_ptr.get().setValue(j)

            predictor_ptr.get().addLabelVector(move(label_vector_ptr))

        cdef ExampleWiseClassificationPredictor predictor = ExampleWiseClassificationPredictor.__new__(
            ExampleWiseClassificationPredictor, num_labels, measure)
        predictor.predictor_ptr = <unique_ptr[IPredictor[uint8]]>move(predictor_ptr)
        return predictor

    def __reduce__(self):
        cdef ExampleWiseClassificationPredictorSerializer serializer = ExampleWiseClassificationPredictorSerializer.__new__(
            ExampleWiseClassificationPredictorSerializer)
        cdef object state = serializer.serialize(self)
        return (ExampleWiseClassificationPredictor, (self.num_labels, self.measure), state)

    def __setstate__(self, state):
        cdef ExampleWiseClassificationPredictorSerializer serializer = ExampleWiseClassificationPredictorSerializer.__new__(
            ExampleWiseClassificationPredictorSerializer)
        serializer.deserialize(self, self.measure, state)


cdef inline unique_ptr[LabelVector] __create_label_vector(list state):
    cdef unique_ptr[LabelVector] label_vector_ptr = make_unique[LabelVector]()
    cdef uint32 num_elements = len(state)
    cdef uint32 i, label_index

    for i in range(num_elements):
        label_index = state[i]
        label_vector_ptr.get().setValue(label_index)

    return move(label_vector_ptr)


cdef class ExampleWiseClassificationPredictorSerializer:
    """
    Allows to serialize and deserialize the label vectors that are stored by a `ExampleWiseClassificationPredictor`.
    """

    cdef __visit_label_vector(self, const LabelVector& label_vector):
        cdef list label_vector_state = []
        cdef LabelVector.index_const_iterator iterator = label_vector.indices_cbegin()
        cdef LabelVector.index_const_iterator end = label_vector.indices_cend()
        cdef uint32 label_index

        while iterator != end:
            label_index = dereference(iterator)
            label_vector_state.append(label_index)
            postincrement(iterator)

        self.state.append(label_vector_state)

    cpdef object serialize(self, ExampleWiseClassificationPredictor predictor):
        """
        Creates and returns a state, which may be serialized using Python's pickle mechanism, from the label vectors
        that are stored by a given `ExampleWiseClassificationPredictor`.

        :param predictor:   The predictor that stores the label vectors to be serialized
        :return:            The state that has been created
        """
        self.state = []
        cdef ExampleWiseClassificationPredictorImpl* predictor_ptr = <ExampleWiseClassificationPredictorImpl*>predictor.predictor_ptr.get()
        predictor_ptr.visit(wrapLabelVectorVisitor(<void*>self, <LabelVectorCythonVisitor>self.__visit_label_vector))
        return (SERIALIZATION_VERSION, self.state)

    cpdef deserialize(self, ExampleWiseClassificationPredictor predictor, object measure, object state):
        """
        Deserializes the label vectors that are stored by a given state and adds them to an
        `ExampleWiseClassificationPredictor`.

        :param predictor:   The predictor, the deserialized rules should be added to
        :param measure:     The measure to be used by the predictor
        :param state:       A state that has previously been created via the function `serialize`
        """
        cdef int version = state[0]

        if version != SERIALIZATION_VERSION:
            raise AssertionError(
                'Version of the serialized predictor is ' + str(version) + ', expected ' + str(SERIALIZATION_VERSION))

        cdef list label_vector_list = state[1]
        cdef uint32 num_label_vectors = len(label_vector_list)
        cdef shared_ptr[IMeasure] measure_ptr = __get_measure_ptr(measure)
        cdef unique_ptr[ExampleWiseClassificationPredictorImpl] predictor_ptr = make_unique[ExampleWiseClassificationPredictorImpl](
            measure_ptr)
        cdef list label_vector_state
        cdef uint32 i

        for i in range(num_label_vectors):
            label_vector_state = label_vector_list[i]
            predictor_ptr.get().addLabelVector(move(__create_label_vector(label_vector_state)))

        predictor.predictor_ptr = <unique_ptr[IPredictor[uint8]]>move(predictor_ptr)
