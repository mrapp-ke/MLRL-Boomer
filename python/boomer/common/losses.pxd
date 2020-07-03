from boomer.common._arrays cimport uint8, uint32, intp, float64


cdef class LabelMatrix:

    # Attributes:

    cdef readonly intp num_examples

    cdef readonly intp num_labels

    # Functions:

    cdef uint8 get_label(self, intp example_index, intp label_index)


cdef class DenseLabelMatrix(LabelMatrix):

    # Attributes:

    cdef uint8[:, ::1] y

    # Functions:

    cdef uint8 get_label(self, intp example_index, intp label_index)


cdef class DefaultPrediction:

    # Attributes:

    cdef float64[::1] predicted_scores


cdef class Prediction(DefaultPrediction):

    # Attributes:

    cdef float64 overall_quality_score


cdef class LabelWisePrediction(Prediction):

    # Attributes:

    cdef float64[::1] quality_scores


cdef class RefinementSearch:

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class DecomposableRefinementSearch(RefinementSearch):

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class NonDecomposableRefinementSearch(RefinementSearch):

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class Loss:

    # Functions:

    cdef DefaultPrediction calculate_default_prediction(self, LabelMatrix label_matrix)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores)
