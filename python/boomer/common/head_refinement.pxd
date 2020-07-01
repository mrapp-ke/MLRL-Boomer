from boomer.common._arrays cimport intp, float64
from boomer.common.losses cimport PredictionSearch, Prediction


cdef class HeadCandidate:

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] predicted_scores

    cdef readonly float64 quality_score


cdef class HeadRefinement:

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices,
                                 PredictionSearch prediction_search, bint uncovered, bint accumulated)

    cdef Prediction calculate_prediction(self, PredictionSearch prediction_search, bint uncovered, bint accumulated)


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices,
                                 PredictionSearch prediction_search, bint uncovered, bint accumulated)

    cdef Prediction calculate_prediction(self, PredictionSearch prediction_search, bint uncovered, bint accumulated)
