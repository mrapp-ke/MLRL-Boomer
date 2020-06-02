from boomer.algorithm._arrays cimport array_float64, fortran_matrix_float64, array_uint8, get_index


DEF _IN = 0
DEF _IP = 1
DEF _RN = 2
DEF _RP = 3


cdef class LabelWiseAveraging(DecomposableCoverageLoss):
    """
    A class for label-wise evaluation.
    """

    def __cinit__(self, Heuristic heuristic):
        self.prediction = LabelIndependentPrediction.__new__(LabelIndependentPrediction)
        self.confusion_matrices_covered = None
        self.heuristic = heuristic

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        cdef intp num_examples = y.shape[0]
        cdef intp num_labels = y.shape[1]
        cdef float64[::1] default_rule = array_float64(num_labels)
        cdef uint8[::1] minority_labels = array_uint8(num_labels)
        cdef float64[::1, :] uncovered_labels = fortran_matrix_float64(num_examples, num_labels)
        cdef float64[::1, :] confusion_matrices_default = fortran_matrix_float64(num_labels, 4)
        cdef float64 threshold = num_examples / 2.0
        cdef float64 sum_uncovered_labels = 0
        cdef uint8 true_label, predicted_label
        cdef intp r, c

        default_rule[:] = 0

        for c in range(num_labels):
            # the default rule predicts the majority-class (label-wise)
            for r in range(num_examples):
                default_rule[c] += y[r, c]

            if default_rule[c] > threshold:
                default_rule[c] = 1
                minority_labels[c] = 0
            else:
                default_rule[c] = 0
                minority_labels[c] = 1

            for r in range(num_examples):
                if default_rule[c] != y[r,c]:
                    sum_uncovered_labels = sum_uncovered_labels + 1


        self.confusion_matrices_default = confusion_matrices_default

        # this stores a matrix which corresponds to the uncovered labels of all examples, where uncovered labels are
        # represented by a one and covered examples are represented by a zero
        uncovered_labels[:,:] = 1

        self.uncovered_labels = uncovered_labels
        self.sum_uncovered_labels = sum_uncovered_labels
        self.minority_labels = minority_labels
        self.true_labels = y

        return default_rule

    cdef void begin_instance_sub_sampling(self):
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        confusion_matrices_default[:, :] = 0

    cdef void update_sub_sample(self, intp example_index, uint32 weight):
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint8[::1, :] true_labels = self.true_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef intp num_labels = minority_labels.shape[0]
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        cdef intp c
        cdef uint8 true_label, predicted_label

        for c in range(num_labels):
            if uncovered_labels[example_index, c] > 0:
                true_label = true_labels[example_index, c]
                predicted_label = minority_labels[c]

                if true_label == 0:
                    if predicted_label == 0:
                        confusion_matrices_default[c, _IN] += weight
                    elif predicted_label == 1:
                        confusion_matrices_default[c, _IP] += weight
                elif true_label == 1:
                    if predicted_label == 0:
                        confusion_matrices_default[c, _RN] += weight
                    elif predicted_label == 1:
                        confusion_matrices_default[c, _RP] += weight

    cdef void remove_from_sub_sample(self, intp example_index, uint32 weight):
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint8[::1, :] true_labels = self.true_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef intp num_labels = minority_labels.shape[0]
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        cdef intp c
        cdef uint8 true_label, predicted_label

        for c in range(num_labels):
            if uncovered_labels[example_index, c] > 0:
                true_label = true_labels[example_index, c]
                predicted_label = minority_labels[c]

                if true_label == 0:
                    if predicted_label == 0:
                        confusion_matrices_default[c, _IN] -= weight
                    elif predicted_label == 1:
                        confusion_matrices_default[c, _IP] -= weight
                elif true_label == 1:
                    if predicted_label == 0:
                        confusion_matrices_default[c, _RN] -= weight
                    elif predicted_label == 1:
                        confusion_matrices_default[c, _RP] -= weight

    cdef void begin_search(self, intp[::1] label_indices):
        cdef LabelIndependentPrediction prediction = self.prediction
        cdef float64[::1] predicted_scores
        cdef float64[::1] quality_scores
        cdef float64[::1, :] confusion_matrices_covered = self.confusion_matrices_covered
        cdef intp num_labels

        if label_indices is None:
            num_labels = self.minority_labels.shape[0]
        else:
            num_labels = label_indices.shape[0]

        if confusion_matrices_covered is None or confusion_matrices_covered.shape[0] != num_labels:
            confusion_matrices_covered = fortran_matrix_float64(num_labels, 4)
            self.confusion_matrices_covered = confusion_matrices_covered
            predicted_scores = array_float64(num_labels)
            prediction.predicted_scores = predicted_scores
            quality_scores = array_float64(num_labels)
            prediction.quality_scores = quality_scores

        confusion_matrices_covered[:, :] = 0
        self.accumulated_confusion_matrices_covered = None
        self.label_indices = label_indices

    cdef void update_search(self, intp example_index, uint32 weight):
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef uint8[::1, :] true_labels = self.true_labels
        cdef float64[::1, :] confusion_matrices_covered = self.confusion_matrices_covered
        cdef intp[::1] label_indices = self.label_indices
        cdef intp num_labels = confusion_matrices_covered.shape[0]
        cdef intp c, l
        cdef uint8 true_label, predicted_label

        for c in range(num_labels):
            l = get_index(c, label_indices)
            if uncovered_labels[example_index, l] > 0:
                true_label = true_labels[example_index, l]
                predicted_label = minority_labels[l]

                if true_label == 0:
                    if predicted_label == 0:
                        confusion_matrices_covered[c, _IN] += weight
                    elif predicted_label == 1:
                        confusion_matrices_covered[c, _IP] += weight
                elif true_label == 1:
                    if predicted_label == 0:
                        confusion_matrices_covered[c, _RN] += weight
                    elif predicted_label == 1:
                        confusion_matrices_covered[c, _RP] += weight

    cdef void reset_search(self):
        cdef float64[::1, :] confusion_matrices_covered = self.confusion_matrices_covered
        cdef intp num_labels = confusion_matrices_covered.shape[0]
        cdef float64[::1, :] accumulated_confusion_matrices_covered = self.accumulated_confusion_matrices_covered
        cdef intp c

        if accumulated_confusion_matrices_covered is None:
            accumulated_confusion_matrices_covered = fortran_matrix_float64(num_labels, 4)
            self.accumulated_confusion_matrices_covered = accumulated_confusion_matrices_covered

            for c in range(num_labels):
                accumulated_confusion_matrices_covered[c, _IN] = confusion_matrices_covered[c, _IN]
                confusion_matrices_covered[c, _IN] = 0
                accumulated_confusion_matrices_covered[c, _IP] = confusion_matrices_covered[c, _IP]
                confusion_matrices_covered[c, _IP] = 0
                accumulated_confusion_matrices_covered[c, _RN] = confusion_matrices_covered[c, _RN]
                confusion_matrices_covered[c, _RN] = 0
                accumulated_confusion_matrices_covered[c, _RP] = confusion_matrices_covered[c, _RP]
                confusion_matrices_covered[c, _RP] = 0
        else:
            for c in range(num_labels):
                accumulated_confusion_matrices_covered[c, _IN] += confusion_matrices_covered[c, _IN]
                confusion_matrices_covered[c, _IN] = 0
                accumulated_confusion_matrices_covered[c, _IP] += confusion_matrices_covered[c, _IP]
                confusion_matrices_covered[c, _IP] = 0
                accumulated_confusion_matrices_covered[c, _RN] += confusion_matrices_covered[c, _RN]
                confusion_matrices_covered[c, _RN] = 0
                accumulated_confusion_matrices_covered[c, _RP] += confusion_matrices_covered[c, _RP]
                confusion_matrices_covered[c, _RP] = 0

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered, bint accumulated):
        cdef LabelIndependentPrediction prediction = self.prediction
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores
        cdef float64 overall_quality_score = 0
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef intp[::1] label_indices = self.label_indices
        cdef float64[::1, :] confusion_matrices_covered = self.accumulated_confusion_matrices_covered if accumulated else self.confusion_matrices_covered
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        cdef intp num_labels = confusion_matrices_covered.shape[0]
        cdef intp c, l
        cdef Heuristic heuristic = self.heuristic

        for c in range(num_labels):
            l = get_index(c, label_indices)
            predicted_scores[c] = <float64> minority_labels[l]
            if uncovered:
                quality_scores[c] = heuristic.evaluate_confusion_matrix(
                    confusion_matrices_default[l, _IN] - confusion_matrices_covered[c, _IN],
                    confusion_matrices_default[l, _IP] - confusion_matrices_covered[c, _IP],
                    confusion_matrices_default[l, _RN] - confusion_matrices_covered[c, _RN],
                    confusion_matrices_default[l, _RP] - confusion_matrices_covered[c, _RP],
                    confusion_matrices_covered[c, _IN],
                    confusion_matrices_covered[c, _IP],
                    confusion_matrices_covered[c, _RN],
                    confusion_matrices_covered[c, _RP]
                )
            else:
                quality_scores[c] = heuristic.evaluate_confusion_matrix(
                    confusion_matrices_covered[c, _IN],
                    confusion_matrices_covered[c, _IP],
                    confusion_matrices_covered[c, _RN],
                    confusion_matrices_covered[c, _RP],
                    confusion_matrices_default[l, _IN] - confusion_matrices_covered[c, _IN],
                    confusion_matrices_default[l, _IP] - confusion_matrices_covered[c, _IP],
                    confusion_matrices_default[l, _RN] - confusion_matrices_covered[c, _RN],
                    confusion_matrices_default[l, _RP] - confusion_matrices_covered[c, _RP],
                )

            overall_quality_score += quality_scores[c]

        prediction.overall_quality_score = overall_quality_score / num_labels

        return prediction

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint8[::1, :] true_labels = self.true_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef float64 sum_uncovered_labels = self.sum_uncovered_labels
        cdef intp num_labels = predicted_scores.shape[0]
        cdef intp c, l

        # Only the labels that are predicted by the new rule must be considered
        for c in range(num_labels):
            l = get_index(c, label_indices)

            if uncovered_labels[example_index, l] == 1:
                uncovered_labels[example_index, l] = 0

                if minority_labels[l] == true_labels[example_index, l]:
                    sum_uncovered_labels = sum_uncovered_labels - 1

        self.sum_uncovered_labels = sum_uncovered_labels