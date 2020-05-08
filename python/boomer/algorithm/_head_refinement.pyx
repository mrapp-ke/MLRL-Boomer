"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for finding the heads of rules.
"""
import numpy as np
from boomer.algorithm._arrays cimport array_intp, array_float64
from boomer.algorithm._utils cimport get_index
from boomer.algorithm._losses cimport LabelIndependentPrediction

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

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered):
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
        :return:                A 'HeadCandidate' that stores information about the head that has been found, if the
                                head is better than `best_head`, None otherwise
        """
        pass

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered):
        """
        Calculates the optimal scores to be predicted by a rule, as well as the rule's overall quality score, given a
        specific loss function.

        The loss function must have been prepared properly via calls to the functions `begin_search` and
        `update_search`.

        :param loss:            The `Loss` to be minimized
        :param uncovered:       0, if the rule for which the optimal scores should be calculated covers all examples
                                that have been provided to the loss function so far, 1, if the rule covers all examples
                                that have not been provided yet
        :return:                A `Prediction` that stores the optimal scores to be predicted by the rule, as well as
                                its overall quality score
        """
        pass


cdef class FullHeadRefinement(HeadRefinement):
    """
    Allows to find the best multi-label head that predicts for all labels.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered):
        cdef Prediction prediction = loss.evaluate_label_dependent_predictions(uncovered)
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

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered):
        cdef Prediction prediction = loss.evaluate_label_dependent_predictions(uncovered)
        return prediction

cdef class PartialHeadRefinement(HeadRefinement):

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered):
        cdef LabelIndependentPrediction prediction = loss.evaluate_label_independent_predictions(uncovered)
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores
        cdef intp num_labels = predicted_scores.shape[0]
        cdef intp num_label_indices
        if label_indices is None:
            num_label_indices = num_labels
        else:
            num_label_indices = label_indices.shape[0]
        cdef float64[::1] candidate_predicted_scores
        cdef HeadCandidate candidate
        cdef intp i, c, c2, c3, l

        cdef intp sorted_label_indices_length = 0
        cdef intp[::1] sorted_indices = array_intp(num_label_indices)

        cdef intp[::1] current_head_candidate = array_intp(num_label_indices)
        cdef intp current_head_candidate_length = 0

        cdef intp[::1] best_head_candidate = array_intp(num_label_indices)
        cdef intp best_head_candidate_length = 0

        cdef float64 best_quality_score, total_quality_score, quality_score
        cdef intp should_continue

        # Insertion sort
        for c in range(0, num_label_indices):
            l = get_index(c, label_indices)
            for c2 in range(0, num_label_indices):
                if c2 >= sorted_label_indices_length or quality_scores[sorted_indices[c2]] > quality_scores[c]:
                    # Shift
                    for c3 in range(sorted_label_indices_length - 1, c2 - 1, -1):
                        sorted_indices[c3 + 1] = sorted_indices[c3]

                    # Insert
                    sorted_indices[c2] = c
                    sorted_label_indices_length += 1

                    break

        for c in range(0, num_labels):
            # select the top element of sorted_label_indices excluding labels already contained

            c2 = 0
            should_continue = True
            no_improvement = False

            while should_continue:
                should_continue = False

                if c2 >= sorted_label_indices_length:
                    no_improvement = True
                    break

                # checks if current_head_candidate contains sorted_label_indices[c2]
                for c3 in range(0, current_head_candidate_length):
                    if current_head_candidate[c3] == get_index(sorted_indices[c2], label_indices):
                        should_continue = True
                        c2 += 1
                        continue

            if no_improvement:
                break

            current_head_candidate[current_head_candidate_length] = sorted_indices[c2]
            current_head_candidate_length += 1

            maximum_lift = 1 # TODO

            for c2 in range(0, current_head_candidate_length):
                total_quality_score += quality_scores[current_head_candidate[c2]]

            total_quality_score /= current_head_candidate_length

            quality_score = total_quality_score * self.lift(total_quality_score, current_head_candidate_length)

            if best_head_candidate_length == 0 or quality_score < best_quality_score:
                best_head_candidate_length = current_head_candidate_length
                # deep copy
                for c2 in range(0, best_head_candidate_length):
                    best_head_candidate[c2] = current_head_candidate[c2]

                best_quality_score = quality_score

            max_score = total_quality_score * maximum_lift

            if max_score < best_quality_score:
                # prunable by decomposition
                break

        if best_head is None or best_quality_score < best_head.quality_score:
            # Create a new `HeadCandidate` and return it...
            candidate_label_indices = array_intp(best_head_candidate_length)
            candidate_predicted_scores = array_float64(best_head_candidate_length)

            for c in range(0, best_head_candidate_length):
                candidate_label_indices[c] = get_index(best_head_candidate[c], label_indices)
                candidate_predicted_scores[c] = predicted_scores[best_head_candidate[c]]

            candidate = HeadCandidate.__new__(HeadCandidate, candidate_label_indices, candidate_predicted_scores,
                                              best_quality_score)
            return candidate

        # Return None, as the quality_score of the found head is worse than that of `best_head`...
        return None

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered):
        cdef Prediction prediction = loss.evaluate_label_dependent_predictions(uncovered)
        return prediction

    cdef float64 lift(self, float64 quality_score, intp labelcount):
        # Example lift function, labelcount only breaks ties between equal scores
        return (quality_score / labelcount) - 0.001 * labelcount

cdef class SingleLabelHeadRefinement(HeadRefinement):
    """
    Allows to find the best single-label head that predicts for a single label.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered):
        cdef LabelIndependentPrediction prediction = loss.evaluate_label_independent_predictions(uncovered)
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

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered):
        cdef Prediction prediction = loss.evaluate_label_independent_predictions(uncovered)
        return prediction
