"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from boomer.common._types cimport uint32
from boomer.common.input cimport CContiguousLabelMatrix, CContiguousLabelMatrixImpl, ILabelMatrix
from boomer.boosting.losses_label_wise cimport LabelWiseLoss
from boomer.boosting.losses_example_wise cimport ExampleWiseLoss

from libcpp.memory cimport shared_ptr, make_unique, dynamic_pointer_cast
from libcpp.utility cimport move


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

    def __cinit__(self, uint32 num_labels):
        """
        :param num_labels: The total number of available labels
        """
        self.num_labels = num_labels

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
            ExampleWiseClassificationPredictor, num_cols)
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
            ExampleWiseClassificationPredictor, num_labels)
        predictor.predictor_ptr = <unique_ptr[IPredictor[uint8]]>move(predictor_ptr)
        return predictor

    def __reduce__(self):
        return (ExampleWiseClassificationPredictor, (self.num_labels,))
