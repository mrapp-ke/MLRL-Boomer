from boomer.common._arrays cimport uint32, intp, float64
from boomer.common.losses cimport LabelMatrix
from boomer.common.losses cimport Loss, RefinementSearch
from boomer.common.losses cimport DefaultPrediction, Prediction, LabelWisePrediction


cdef class CoverageLoss(Loss):

    # Attributes:

    cdef readonly float64 sum_uncovered_labels

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores)
