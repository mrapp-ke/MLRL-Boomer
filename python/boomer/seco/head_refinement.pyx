from boomer.common._arrays cimport intp, float64
from boomer.common._predictions cimport LabelWisePredictionCandidate
from boomer.common._tuples cimport IndexedFloat64, compareIndexedFloat64
from boomer.seco.lift_functions cimport LiftFunction

from libc.stdlib cimport qsort, malloc, realloc, free


cdef class PartialHeadRefinement(HeadRefinement):

    def __cinit__(self, LiftFunction lift_function):
        self.lift_function_ptr = lift_function.lift_function_ptr

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil:
        cdef LabelWisePredictionCandidate* prediction = refinement_search.calculateLabelWisePrediction(uncovered,
                                                                                                       accumulated)
        cdef float64* predicted_scores = prediction.predictedScores_
        cdef float64* quality_scores = prediction.qualityScores_
        cdef intp num_predictions = prediction.numPredictions_
        cdef float64* candidate_predicted_scores
        cdef uint32* candidate_label_indices
        cdef uint32* sorted_indices = NULL
        cdef intp best_head_candidate_length = 0
        cdef float64 best_quality_score, total_quality_score = 0, quality_score, maximum_lift, max_score
        cdef intp c

        cdef AbstractLiftFunction* lift_function = self.lift_function_ptr.get()

        try:
            if label_indices == NULL:
                sorted_indices = __argsort(quality_scores, num_predictions)

                maximum_lift = lift_function.getMaxLift()
                for c in range(num_predictions):
                    # select the top element of sorted_label_indices excluding labels already contained
                    total_quality_score += 1 - quality_scores[sorted_indices[c]]

                    quality_score = 1 - (total_quality_score / (c + 1)) * lift_function.calculateLift(c + 1)


                    if best_head_candidate_length == 0 or quality_score < best_quality_score:
                        best_head_candidate_length = c + 1

                        best_quality_score = quality_score

                    max_score = quality_score * maximum_lift

                    if max_score < best_quality_score:
                        # prunable by decomposition
                        break
            else:
                for c in range(num_predictions):
                    # select the top element of sorted_label_indices excluding labels already contained
                    total_quality_score += 1 - quality_scores[c]

                best_quality_score = 1 - (total_quality_score / num_predictions) * lift_function.calculateLift(num_predictions)

                best_head_candidate_length = num_predictions

            if best_head == NULL or best_quality_score < best_head.overallQualityScore_:
                if recyclable_head == NULL:
                    # Create a new `PredictionCandidate` and return it...
                    candidate_label_indices = <uint32*>malloc(best_head_candidate_length * sizeof(uint32))
                    candidate_predicted_scores = <float64*>malloc(best_head_candidate_length * sizeof(float64))

                    if label_indices == NULL:
                        for c in range(best_head_candidate_length):
                            candidate_label_indices[c] = sorted_indices[c] if label_indices == NULL else label_indices[sorted_indices[c]]
                            candidate_predicted_scores[c] = predicted_scores[sorted_indices[c]]
                    else:
                        for c in range(best_head_candidate_length):
                            candidate_label_indices[c] = label_indices[c]
                            candidate_predicted_scores[c] = predicted_scores[c]

                    return new PredictionCandidate(best_head_candidate_length, candidate_label_indices,
                                                   candidate_predicted_scores, best_quality_score)
                else:
                    # Modify the `recyclable_head` and return it...
                    candidate_label_indices = recyclable_head.labelIndices_
                    candidate_predicted_scores = recyclable_head.predictedScores_

                    if recyclable_head.numPredictions_ != best_head_candidate_length:
                        recyclable_head.numPredictions_ = best_head_candidate_length
                        candidate_label_indices = <uint32*>realloc(candidate_label_indices, best_head_candidate_length * sizeof(uint32))
                        recyclable_head.labelIndices_ = candidate_label_indices
                        candidate_predicted_scores = <float64*>realloc(candidate_predicted_scores, best_head_candidate_length * sizeof(float64))
                        recyclable_head.predictedScores_ = candidate_predicted_scores

                    if label_indices == NULL:
                        for c in range(best_head_candidate_length):
                            candidate_label_indices[c] = sorted_indices[c] if label_indices == NULL else label_indices[sorted_indices[c]]
                            candidate_predicted_scores[c] = predicted_scores[sorted_indices[c]]
                    else:
                        for c in range(best_head_candidate_length):
                            candidate_label_indices[c] = label_indices[c]
                            candidate_predicted_scores[c] = predicted_scores[c]

                    recyclable_head.overallQualityScore_ = best_quality_score
                    return recyclable_head

            # Return NULL, as the quality_score of the found head is worse than that of `best_head`...
            return NULL
        finally:
            free(sorted_indices)

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil:
        return refinement_search.calculateLabelWisePrediction(uncovered, accumulated)


cdef inline uint32* __argsort(float64* a, intp num_elements) nogil:
    """
    Creates and returns an array that stores the indices of the elements in a given array when sorted in ascending 
    order.

    :param a:               A pointer to an array of type float64, shape `(num_elements)`, representing the values of
                            the array to be sorted
    :param num_elements:    The number of elements in the array `a`
    :return:                A pointer to an array of type `uint32`, shape `(num_elements)`, representing the indices of
                            the values in the given array when sorted in ascending order
    """
    cdef IndexedFloat64* tmp_array = <IndexedFloat64*>malloc(num_elements * sizeof(IndexedFloat64))
    cdef uint32* sorted_array = <uint32*>malloc(num_elements * sizeof(uint32))
    cdef uint32 i

    try:
        for i in range(num_elements):
            tmp_array[i].index = i
            tmp_array[i].value = a[i]

        qsort(tmp_array, num_elements, sizeof(IndexedFloat64), &compareIndexedFloat64)

        for i in range(num_elements):
            sorted_array[i] = tmp_array[i].index
    finally:
        free(tmp_array)

    return sorted_array
