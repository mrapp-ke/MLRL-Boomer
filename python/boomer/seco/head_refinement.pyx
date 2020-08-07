from boomer.common._arrays cimport float64, array_intp, array_float64, get_index
from boomer.common._tuples cimport IndexedFloat64, compare_indexed_float64
from boomer.common.rule_evaluation cimport LabelWisePrediction
from boomer.seco.lift_functions cimport LiftFunction

from libc.stdlib cimport qsort, malloc, realloc, free


cdef class PartialHeadRefinement(HeadRefinement):

    def __cinit__(self, LiftFunction lift_function):
        self.lift_function_ptr = lift_function.lift_function_ptr

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, HeadCandidate* recyclable_head,
                                  intp[::1] label_indices, AbstractRefinementSearch* refinement_search, bint uncovered,
                                  bint accumulated) nogil:
        cdef LabelWisePrediction* prediction = refinement_search.calculateLabelWisePrediction(uncovered, accumulated)
        cdef float64* predicted_scores = prediction.predictedScores_
        cdef float64* quality_scores = prediction.qualityScores_
        cdef intp num_predictions
        cdef float64* candidate_predicted_scores
        cdef intp* candidate_label_indices
        cdef intp* sorted_indices
        cdef intp best_head_candidate_length = 0
        cdef float64 best_quality_score, total_quality_score = 0, quality_score, maximum_lift, max_score
        cdef intp c

        cdef AbstractLiftFunction* lift_function = self.lift_function_ptr.get()

        if label_indices is None:
            num_predictions = prediction.numPredictions_

            sorted_indices = __argsort(quality_scores, num_predictions)

            maximum_lift = lift_function.getMaxLift()
            for c in range(0, num_predictions):
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

            free(sorted_indices)
        else:
            num_predictions = label_indices.shape[0]

            for c in range(0, num_predictions):
                # select the top element of sorted_label_indices excluding labels already contained
                total_quality_score += 1 - quality_scores[c]

            best_quality_score = 1 - (total_quality_score / num_predictions) * lift_function.calculateLift(num_predictions)

            best_head_candidate_length = label_indices.shape[0]

        if best_head == NULL or best_quality_score < best_head.qualityScore_:
            if recyclable_head == NULL:
                # Create a new `HeadCandidate` and return it...
                candidate_label_indices = <intp*>malloc(best_head_candidate_length * sizeof(intp))
                candidate_predicted_scores = <float64*>malloc(best_head_candidate_length * sizeof(float64))

                if label_indices is None:
                    for c in range(0, best_head_candidate_length):
                        candidate_label_indices[c] = get_index(sorted_indices[c], label_indices)
                        candidate_predicted_scores[c] = predicted_scores[sorted_indices[c]]
                else:
                    for c in range(0, best_head_candidate_length):
                        candidate_label_indices[c] = label_indices[c]
                        candidate_predicted_scores[c] = predicted_scores[c]

                return new HeadCandidate(best_head_candidate_length, candidate_label_indices,
                                         candidate_predicted_scores, best_quality_score)
            else:
                candidate_label_indices = recyclable_head.labelIndices_
                candidate_predicted_scores = recyclable_head.predictedScores_

                if recyclable_head.numPredictions_ != best_head_candidate_length:
                    recyclable_head.numPredictions_ = best_head_candidate_length
                    candidate_label_indices = <intp*>realloc(candidate_label_indices, best_head_candidate_length * sizeof(intp))
                    recyclable_head.labelIndices_ = candidate_label_indices
                    candidate_predicted_scores = <float64*>realloc(candidate_predicted_scores, best_head_candidate_length * sizeof(float64))
                    recyclable_head.predictedScores_ = candidate_predicted_scores

                # Modify the `recyclable_head` and return it...
                if label_indices is None:
                    for c in range(best_head_candidate_length):
                        candidate_label_indices[c] = get_index(sorted_indices[c], label_indices)
                        candidate_predicted_scores[c] = predicted_scores[sorted_indices[c]]
                else:
                    for c in range(best_head_candidate_length):
                        candidate_label_indices[c] = label_indices[c]
                        candidate_predicted_scores[c] = predicted_scores[c]

                recyclable_head.qualityScore_ = best_quality_score
                return recyclable_head

        # Return NULL, as the quality_score of the found head is worse than that of `best_head`...
        return NULL

    cdef Prediction* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                          bint accumulated) nogil:
        cdef Prediction* prediction = refinement_search.calculateLabelWisePrediction(uncovered, accumulated)
        return prediction


cdef inline intp* __argsort(float64* a, intp num_elements) nogil:
    """
    Creates and returns an array that stores the indices of the elements in a given array when sorted in ascending 
    order.

    :param a:               A pointer to an array of type float64, shape `(num_elements)`, representing the values of
                            the array to be sorted
    :param num_elements:    The number of elements in the array `a`
    :return:                A pointer to an array of type `intp`, shape `(num_elements)`, representing the indices of
                            the values in the given array when sorted in ascending order
    """
    cdef IndexedFloat64* tmp_array = <IndexedFloat64*>malloc(num_elements * sizeof(IndexedFloat64))
    cdef intp* sorted_array = <intp*>malloc(num_elements * sizeof(intp))
    cdef intp i

    try:
        for i in range(num_elements):
            tmp_array[i].index = i
            tmp_array[i].value = a[i]

        qsort(tmp_array, num_elements, sizeof(IndexedFloat64), &compare_indexed_float64)

        for i in range(num_elements):
            sorted_array[i] = tmp_array[i].index
    finally:
        free(tmp_array)

    return sorted_array
