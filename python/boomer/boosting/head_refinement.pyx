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

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const intp* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil:
        cdef PredictionCandidate* prediction = refinement_search.calculateExampleWisePrediction(uncovered, accumulated)
        cdef intp num_predictions = prediction.numPredictions_
        cdef float64* predicted_scores = prediction.predictedScores_
        cdef float64 overall_quality_score = prediction.overallQualityScore_
        cdef intp* candidate_label_indices = NULL
        cdef float64* candidate_predicted_scores
        cdef intp c

        # The quality score must be better than that of `best_head`...
        if best_head == NULL or overall_quality_score < best_head.overallQualityScore_:
            if recyclable_head == NULL:
                # Create a new `PredictionCandidate` and return it...
                candidate_predicted_scores = <float64*>malloc(num_predictions * sizeof(float64))

                for c in range(num_predictions):
                    candidate_predicted_scores[c] = predicted_scores[c]

                if label_indices != NULL:
                    candidate_label_indices = <intp*>malloc(num_predictions * sizeof(intp))

                    for c in range(num_predictions):
                        candidate_label_indices[c] = label_indices[c]

                return new PredictionCandidate(num_predictions, candidate_label_indices, candidate_predicted_scores,
                                               overall_quality_score)
            else:
                # Modify the `recyclable_head` and return it...
                for c in range(num_predictions):
                    best_head.predictedScores_[c] = predicted_scores[c]

                best_head.overallQualityScore_ = overall_quality_score
                return best_head

        # Return NULL, as the quality score of the found head is worse than that of `best_head`...
        return NULL

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil:
        return refinement_search.calculateExampleWisePrediction(uncovered, accumulated)
