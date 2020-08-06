"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for finding the heads of rules.
"""
from boomer.common._arrays cimport array_intp, array_float64, get_index
from boomer.common.rule_evaluation cimport LabelWisePrediction

from libc.stdlib cimport malloc


cdef class HeadRefinement:
    """
    A base class for all classes that allow to find the best single- or multi-label head for a rule.
    """

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, HeadCandidate* recyclable_head,
                                  intp[::1] label_indices, RefinementSearch refinement_search, bint uncovered,
                                  bint accumulated) nogil:
        """
        Finds and returns the best head for a rule given the predictions that are provided by a `RefinementSearch`.

        The `RefinementSearch` must have been prepared properly via calls to the function
        `RefinementSearch#update_search`.

        :param best_head:           A pointer to an instance of the C++ class `HeadCandidate` that corresponds to the
                                    best rule known so far (as found in the previous or current refinement iteration) or
                                    NULL, if no such rule is available yet. The new head must be better than this one,
                                    otherwise it is discarded
        :param recyclable_head:     A pointer to an instance of the C++ class `HeadCandidate` that may be modified
                                    instead of creating a new instance to avoid unnecessary memory allocations or NULL,
                                    if no such instance is available
        :param label_indices:       An array of dtype int, shape `(num_labels)`, representing the indices of the labels
                                    for which the head may predict or None, if the head may predict for all labels
        :param refinement_search:   The `RefinementSearch` that should be used
        :param uncovered:           0, if the rule for which the head should be found covers all examples that have been
                                    provided to the `RefinementSearch` so far, 1, if the rule covers all examples that
                                    have not been provided yet
        :param accumulated:         0, if the rule covers all examples that have been provided since the
                                    `RefinementSearch` has been reset for the last time, 1, if the rule covers all
                                    examples that have been provided so far
        :return:                    A pointer to an instance of the C++ class 'HeadCandidate' that stores information
                                    about the head that has been found, if the head is better than `best_head`, NULL
                                    otherwise
        """
        pass

    cdef Prediction* calculate_prediction(self, RefinementSearch refinement_search, bint uncovered,
                                          bint accumulated) nogil:
        """
        Calculates the optimal scores to be predicted by a rule, as well as the rule's overall quality score, using a
        `RefinementSearch`.

        The `RefinementSearch` must have been prepared properly via calls to the function
        `RefinementSearch#update_search`.

        :param refinement_search:   The `RefinementSearch` that should be used
        :param uncovered:           0, if the rule for which the optimal scores should be calculated covers all examples
                                    that have been provided to the `RefinementSearch` so far, 1, if the rule covers all
                                    examples that have not been provided yet
        :param accumulated          0, if the rule covers all examples that have been provided since the
                                    `RefinementSearch` has been reset for the last time, 1, if the rule covers all
                                    examples that have been  provided so far
        :return:                    A pointer to an object of type `Prediction` that stores the optimal scores to be
                                    predicted by the rule, as well as its overall quality score
        """
        pass


cdef class SingleLabelHeadRefinement(HeadRefinement):
    """
    Allows to find the best single-label head that predicts for a single label.
    """

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, HeadCandidate* recyclable_head,
                                  intp[::1] label_indices, RefinementSearch refinement_search, bint uncovered,
                                  bint accumulated) nogil:
        cdef LabelWisePrediction* prediction = refinement_search.calculate_label_wise_prediction(uncovered, accumulated)
        cdef intp num_predictions = prediction.numPredictions_
        cdef float64* predicted_scores = prediction.predictedScores_
        cdef float64* quality_scores = prediction.qualityScores_
        cdef intp best_c = 0
        cdef float64 best_quality_score = quality_scores[best_c]
        cdef intp* candidate_label_indices
        cdef float64* candidate_predicted_scores
        cdef float64 quality_score
        cdef intp c

        # Find the best single-label head...
        for c in range(1, num_predictions):
            quality_score = quality_scores[c]

            if quality_score < best_quality_score:
                best_quality_score = quality_score
                best_c = c

        # The quality score must be better than that of `best_head`...
        if best_head == NULL or best_quality_score < best_head.qualityScore_:
            if recyclable_head == NULL:
                # Create a new `HeadCandidate` and return it...
                candidate_label_indices = <intp*>malloc(sizeof(intp))
                candidate_label_indices[0] = get_index(best_c, label_indices)
                candidate_predicted_scores = <float64*>malloc(sizeof(float64))
                candidate_predicted_scores[0] = predicted_scores[best_c]
                return new HeadCandidate(1, candidate_label_indices, candidate_predicted_scores, best_quality_score)
            else:
                # Modify the `recyclable_head` and return it...
                recyclable_head.labelIndices_[0] = get_index(best_c, label_indices)
                recyclable_head.predictedScores_[0] = predicted_scores[best_c]
                recyclable_head.qualityScore_ = best_quality_score
                return recyclable_head

        # Return NULL, as the quality_score of the found head is worse than that of `best_head`...
        return NULL

    cdef Prediction* calculate_prediction(self, RefinementSearch refinement_search, bint uncovered,
                                          bint accumulated) nogil:
        cdef Prediction* prediction = refinement_search.calculate_label_wise_prediction(uncovered, accumulated)
        return prediction
