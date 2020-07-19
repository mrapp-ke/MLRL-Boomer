"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for finding the heads of rules, which are specific to boosting algorithms.
"""
from boomer.common._arrays cimport float64, array_float64

from libc.stdlib cimport malloc


cdef class FullHeadRefinement(HeadRefinement):
    """
    Allows to find the best multi-label head that predicts for all labels.
    """

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, intp[::1] label_indices,
                                  RefinementSearch refinement_search, bint uncovered, bint accumulated):
        cdef Prediction* prediction = refinement_search.calculate_example_wise_prediction(uncovered, accumulated)
        cdef intp num_predictions = prediction.numPredictions_
        cdef float64* predicted_scores = prediction.predictedScores_
        cdef float64 overall_quality_score = prediction.overallQualityScore_
        cdef intp* candidate_label_indices = NULL
        cdef float64* candidate_predicted_scores
        cdef HeadCandidate* candidate
        cdef intp c

        if best_head == NULL:
            # Create a new `HeadCandidate` and return it...
            candidate_predicted_scores = <float64*>malloc(num_predictions * sizeof(float64))

            for c in range(num_predictions):
                candidate_predicted_scores[c] = predicted_scores[c]

            if label_indices is not None:
                candidate_label_indices = <intp*>malloc(num_predictions * sizeof(intp))

                for c in range(num_predictions):
                    candidate_label_indices[c] = label_indices[c]

            candidate = new HeadCandidate(num_predictions, candidate_label_indices, candidate_predicted_scores,
                                          overall_quality_score)
            return candidate
        else:
            # The quality score must be better than that of `best_head`...
            if overall_quality_score < best_head.qualityScore_:
                # Modify the `best_head` and return it...
                for c in range(num_predictions):
                    best_head.predictedScores_[c] = predicted_scores[c]

                best_head.qualityScore_ = overall_quality_score
                return best_head

        # Return NULL, as the quality score of the found head is worse than that of `best_head`...
        return NULL

    cdef Prediction* calculate_prediction(self, RefinementSearch refinement_search, bint uncovered, bint accumulated):
        cdef Prediction* prediction = refinement_search.calculate_example_wise_prediction(uncovered, accumulated)
        return prediction
