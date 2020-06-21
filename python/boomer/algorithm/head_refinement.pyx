"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for finding the heads of rules.
"""
from boomer.algorithm._arrays cimport array_intp, array_float64, get_index
from boomer.algorithm.losses cimport LabelIndependentPrediction
from boomer.algorithm.lift_functions cimport LiftFunction

from libc.stdlib cimport qsort

from cpython.mem cimport PyMem_Malloc as malloc, PyMem_Free as free


cdef class HeadCandidate:
    """
    Stores information about a potential head of a rule.
    """

    def __cinit__(self, intp[::1] label_indices, float64[::1] predicted_scores, float64 quality_score):
        """
        :param label_indices:       An array of dtype int, shape `(num_predicted_labels)`, representing the indices of
                                    the labels for which the head predicts or None, if the head predicts for all labels
        :param predicted_scores:    An array of dtype float, shape `(num_predicted_labels)`, representing the scores
                                    that are predicted by the head. The predicted scores correspond to the indices in
                                    the array `label_indices`.  If `label_indices` is None, the scores correspond to all
                                    labels in the training data
        :param quality_score:       A score that measures the quality of the head
        """
        self.label_indices = label_indices
        self.predicted_scores = predicted_scores
        self.quality_score = quality_score


cdef class HeadRefinement:
    """
    A base class for all classes that allow to find the best single- or multi-label head for a rule.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated):
        """
        Finds and returns the best head for a rule given a specific loss function.

        The loss function must have been prepared properly via calls to the functions `begin_search` and
        `update_search`.

        :param best_head:       The `HeadCandidate` that corresponds to the best rule known so far (as found in the
                                previous or current refinement iteration) or None, if no such rule is available yet. The
                                new head must be better than this one, otherwise it is discarded. If the new head is
                                better, this `HeadCandidate` will be modified accordingly instead of creating a new
                                instance to avoid unnecessary memory allocations
        :param label_indices:   An array of dtype int, shape `(num_labels)`, representing the indices of the labels for
                                which the head may predict or None, if the head may predict for all labels
        :param loss:            The `Loss` to be minimized
        :param uncovered:       0, if the rule for which the head should be found covers all examples that have been
                                provided to the loss function so far, 1, if the rule covers all examples that have not
                                been provided yet
        :param accumulated:     0, if the rule covers all examples that have been provided since the loss function has
                                been reset for the last time, 1, if the rule covers all examples that have been provided
                                so far
        :return:                A 'HeadCandidate' that stores information about the head that has been found, if the
                                head is better than `best_head`, None otherwise
        """
        pass

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated):
        """
        Calculates the optimal scores to be predicted by a rule, as well as the rule's overall quality score, given a
        specific loss function.

        The loss function must have been prepared properly via calls to the functions `begin_search` and
        `update_search`.

        :param loss:            The `Loss` to be minimized
        :param uncovered:       0, if the rule for which the optimal scores should be calculated covers all examples
                                that have been provided to the loss function so far, 1, if the rule covers all examples
                                that have not been provided yet
        :param accumulated      0, if the rule covers all examples that have been provided since the loss function has
                                been reset for the last time, 1, if the rule covers all examples that have been provided
                                so far
        :return:                A `Prediction` that stores the optimal scores to be predicted by the rule, as well as
                                its overall quality score
        """
        pass


cdef class FullHeadRefinement(HeadRefinement):
    """
    Allows to find the best multi-label head that predicts for all labels.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated):
        cdef Prediction prediction = loss.evaluate_label_dependent_predictions(uncovered, accumulated)
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64 overall_quality_score = prediction.overall_quality_score
        cdef intp num_labels = predicted_scores.shape[0]
        cdef float64[::1] candidate_predicted_scores
        cdef HeadCandidate candidate
        cdef intp c

        if best_head is None:
            # Create a new `HeadCandidate` and return it...
            candidate_predicted_scores = array_float64(num_labels)

            for c in range(num_labels):
                candidate_predicted_scores[c] = predicted_scores[c]

            candidate = HeadCandidate.__new__(HeadCandidate, label_indices, candidate_predicted_scores,
                                              overall_quality_score)
            return candidate
        else:
            # The quality score must be better than that of `best_head`...
            if overall_quality_score < best_head.quality_score:
                # Modify the `best_head` and return it...
                for c in range(num_labels):
                    best_head.predicted_scores[c] = predicted_scores[c]

                best_head.quality_score = overall_quality_score
                return best_head

        # Return None, as the quality score of the found head is worse than that of `best_head`...
        return None

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated):
        cdef Prediction prediction = loss.evaluate_label_dependent_predictions(uncovered, accumulated)
        return prediction

cdef class PartialHeadRefinement(HeadRefinement):

    def __cinit__(self, LiftFunction lift):
        self.lift = lift

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated):
        cdef LabelIndependentPrediction prediction = loss.evaluate_label_independent_predictions(uncovered, accumulated)
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

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated):
        cdef Prediction prediction = loss.evaluate_label_independent_predictions(uncovered, accumulated)
        return prediction

cdef class SingleLabelHeadRefinement(HeadRefinement):
    """
    Allows to find the best single-label head that predicts for a single label.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated):
        cdef LabelIndependentPrediction prediction = loss.evaluate_label_independent_predictions(uncovered, accumulated)
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores
        cdef intp num_labels = predicted_scores.shape[0]
        cdef intp best_c = 0
        cdef float64 best_quality_score = quality_scores[best_c]
        cdef HeadCandidate candidate
        cdef intp[::1] candidate_label_indices
        cdef float64[::1] candidate_predicted_scores
        cdef float64 quality_score
        cdef intp c

        # Find the best single-label head...
        for c in range(1, num_labels):
            quality_score = quality_scores[c]

            if quality_score < best_quality_score:
                best_quality_score = quality_score
                best_c = c

        if best_head is None:
            # Create a new `HeadCandidate` and return it...
            candidate_label_indices = array_intp(1)
            candidate_label_indices[0] = get_index(best_c, label_indices)
            candidate_predicted_scores = array_float64(1)
            candidate_predicted_scores[0] = predicted_scores[best_c]
            candidate = HeadCandidate.__new__(HeadCandidate, candidate_label_indices, candidate_predicted_scores,
                                              best_quality_score)
            return candidate
        else:
            # The quality score must be better than that of `best_head`...
            if best_quality_score < best_head.quality_score:
                best_head.label_indices[0] = get_index(best_c, label_indices)
                best_head.predicted_scores[0] = predicted_scores[best_c]
                best_head.quality_score = best_quality_score
                return best_head

        # Return None, as the quality_score of the found head is worse than that of `best_head`...
        return None

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated):
        cdef Prediction prediction = loss.evaluate_label_independent_predictions(uncovered, accumulated)
        return prediction


cdef inline intp[::1] __argsort(float64[::1] values):
    """
    Sorts the indices of an float64-array in ascending order
    
    :param values:         An array of dtype float, shape `(num_examples)`, representing the values of the array to
                           argsort
    :param sorted_array:   An array of dtype intp, shape `(num_examples)`, to store the sorted indices
    :return:               A pointer to a C-array of type intp, representing the sorted indices of the array
    """
    cdef intp num_values = values.shape[0]
    cdef IndexedValue* tmp_array = <IndexedValue*>malloc(num_values * sizeof(IndexedValue))
    cdef intp[::1] sorted_array = array_intp(num_values)
    cdef intp i

    try:
        for i in range(num_values):
            tmp_array[i].index = i
            tmp_array[i].value = values[i]

        qsort(tmp_array, num_values, sizeof(IndexedValue), &__compare_indexed_value)

        for i in range(num_values):
            sorted_array[i] = tmp_array[i].index
    finally:
        free(tmp_array)

    return sorted_array


cdef int __compare_indexed_value(const void* a, const void* b) nogil:
    """
    Compares the values of two structs of type `IndexedValue`.

    :param a:   A pointer to the first struct
    :param b:   A pointer to the second struct
    :return:    -1 if the value of the first struct is smaller than the value of the second struct, 0 if both values are
                equal, or 1 if the value of the first struct is greater than the value of the second struct
    """
    cdef float64 v1 = (<IndexedValue*>a).value
    cdef float64 v2 = (<IndexedValue*>b).value
    return -1 if v1 < v2 else (0 if v1 == v2 else 1)
