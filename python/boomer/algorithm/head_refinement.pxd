from boomer.algorithm._arrays cimport intp, float64
from boomer.algorithm.losses cimport Loss, Prediction

from libc.math cimport abs


# The minimum difference between two quality scores to be considered unequal
DEF QUALITY_SCORE_THRESHOLD = 0.0000001


cdef inline intp compare_quality_scores(float64 quality_score1, float64 quality_score2):
    """
    Compares two quality scores with each other.

    :param quality_score1:  The first quality score
    :param quality_score2:  The second quality score
    :return:                0, if both quality scores are considered equal, 1 if the first quality score is better, -1
                            if the first quality score is worse
    """
    if abs(quality_score1 - quality_score2) < QUALITY_SCORE_THRESHOLD:
        return 0
    elif quality_score1 < quality_score2:
        return 1
    else:
        return -1


cdef class HeadCandidate:

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] predicted_scores

    cdef readonly float64 quality_score


cdef class HeadRefinement:

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered)


cdef class FullHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered)


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered)
