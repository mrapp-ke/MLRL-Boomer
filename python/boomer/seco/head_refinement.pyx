from boomer.common._arrays cimport float64, array_intp, array_float64, get_index
from boomer.common._tuples cimport IndexedFloat64, compare_indexed_float64
from boomer.common.losses cimport LabelWisePrediction

from libc.stdlib cimport qsort

from cpython.mem cimport PyMem_Malloc as malloc, PyMem_Free as free


cdef class PartialHeadRefinement(HeadRefinement):

    def __cinit__(self, LiftFunction lift):
        self.lift = lift

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices,
                                 RefinementSearch refinement_search, bint uncovered, bint accumulated):
        cdef LabelWisePrediction prediction = refinement_search.calculate_label_wise_prediction(uncovered, accumulated)
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores
        cdef intp num_labels
        cdef HeadCandidate candidate
        cdef float64[::1] candidate_predicted_scores
        cdef intp[::1] candidate_label_indices
        cdef intp[::1] sorted_indices
        cdef intp best_head_candidate_length = 0
        cdef float64 best_quality_score, total_quality_score = 0, quality_score, maximum_lift
        cdef intp c

        cdef LiftFunction lift = self.lift

        if label_indices is None:
            num_labels = predicted_scores.shape[0]

            sorted_indices = __argsort(quality_scores)

            maximum_lift = lift.get_max_lift()
            for c in range(0, num_labels):
                # select the top element of sorted_label_indices excluding labels already contained
                total_quality_score += quality_scores[sorted_indices[c]]

                quality_score = (1 - (1 - total_quality_score) * lift.eval(c + 1)) / (c + 1)

                if best_head_candidate_length == 0 or quality_score < best_quality_score:
                    best_head_candidate_length = c + 1

                    best_quality_score = quality_score

                max_score = quality_score * maximum_lift

                if max_score < best_quality_score:
                    # prunable by decomposition
                    break
        else:
            num_labels = label_indices.shape[0]

            for c in range(0, num_labels):
                # select the top element of sorted_label_indices excluding labels already contained
                total_quality_score += quality_scores[c]

            best_quality_score = (1 - (1 - total_quality_score) * lift.eval(num_labels)) / num_labels

            best_head_candidate_length = label_indices.shape[0]

        if best_head is None:
            # Create a new `HeadCandidate` and return it...
            candidate_label_indices = array_intp(best_head_candidate_length)
            candidate_predicted_scores = array_float64(best_head_candidate_length)

            if label_indices is None:
                for c in range(0, best_head_candidate_length):
                    candidate_label_indices[c] = get_index(sorted_indices[c], label_indices)
                    candidate_predicted_scores[c] = predicted_scores[sorted_indices[c]]
            else:
                for c in range(0, best_head_candidate_length):
                    candidate_label_indices[c] = label_indices[c]
                    candidate_predicted_scores[c] = predicted_scores[c]


            candidate = HeadCandidate.__new__(HeadCandidate, candidate_label_indices, candidate_predicted_scores,
                                              best_quality_score)
            return candidate
        elif best_quality_score < best_head.quality_score:
            if best_head.label_indices.shape[0] != best_head_candidate_length:
                best_head.label_indices = array_intp(best_head_candidate_length)
                best_head.predicted_scores = array_float64(best_head_candidate_length)

            # Modify the `best_head` and return it...
            if label_indices is None:
                for c in range(best_head_candidate_length):
                    best_head.label_indices[c] = get_index(sorted_indices[c], label_indices)
                    best_head.predicted_scores[c] = predicted_scores[sorted_indices[c]]
            else:
                for c in range(best_head_candidate_length):
                    best_head.label_indices[c] = label_indices[c]
                    best_head.predicted_scores[c] = predicted_scores[c]

            best_head.quality_score = best_quality_score
            return best_head
        else:
            # Return None, as the quality_score of the found head is worse than that of `best_head`...
            return None

    cdef Prediction calculate_prediction(self, RefinementSearch refinement_search, bint uncovered, bint accumulated):
        cdef Prediction prediction = refinement_search.calculate_label_wise_prediction(uncovered, accumulated)
        return prediction


cdef inline intp[::1] __argsort(float64[::1] values):
    """
    Creates and returns an array that stores the indices of the elements in a given array when sorted in ascending 
    order.

    :param values:  An array of dtype float, shape `(num_elements)`, representing the values of the array to be sorted
    :return:        An array of dtype int, shape `(num_elements)`, representing the indices of the values in the given 
                    array when sorted in ascending order
    """
    cdef intp num_values = values.shape[0]
    cdef IndexedFloat64* tmp_array = <IndexedFloat64*>malloc(num_values * sizeof(IndexedFloat64))
    cdef intp[::1] sorted_array = array_intp(num_values)
    cdef intp i

    try:
        for i in range(num_values):
            tmp_array[i].index = i
            tmp_array[i].value = values[i]

        qsort(tmp_array, num_values, sizeof(IndexedFloat64), &compare_indexed_float64)

        for i in range(num_values):
            sorted_array[i] = tmp_array[i].index
    finally:
        free(tmp_array)

    return sorted_array
