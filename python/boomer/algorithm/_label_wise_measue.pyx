from boomer.algorithm._arrays cimport uint8, uint32, intp, float64, array_float64, matrix_float64, array_uint32
from boomer.algorithm._losses cimport Loss, LabelIndependentPrediction, Prediction
from boomer.algorithm._utils cimport get_index

DEF _IN = 0
DEF _IP = 1
DEF _RN = 2
DEF _RP = 3

cdef class LabelWiseMeasure(Loss):
    """
    A class for label-wise evaluation.
    """

    def __cinit__(self):
        self.prediction = LabelIndependentPrediction()
        self.confusion_matrices_covered = None
        self.label_indices = None

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        cdef intp num_examples = y.shape[0]
        cdef intp num_labels = y.shape[1]
        cdef float64[::1] default_rule = array_float64(num_labels)
        cdef uint32[::1] minority_labels = array_uint32(num_labels)
        cdef float64[::1, :] uncovered_labels = matrix_float64(num_examples, num_labels)
        cdef float64[::1, :] confusion_matrices_covered = matrix_float64(num_labels, 4)
        cdef float64[::1, :] confusion_matrices_default = matrix_float64(num_labels, 4)
        cdef float64 threshold = num_examples / 2.0
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

            # the confusion_matrix_default is the confusion matrix of the default rule
            for r in range(num_examples):
                true_label = y[r, c]
                predicted_label = <int> default_rule[c]

                if true_label == 0:
                    if predicted_label == 0:
                        confusion_matrices_default[c, _IN] += 1
                    elif predicted_label == 1:
                        confusion_matrices_default[c, _IP] += 1
                elif true_label == 1:
                    if predicted_label == 0:
                        confusion_matrices_default[c, _RN] += 1
                    elif predicted_label == 1:
                        confusion_matrices_default[c, _RP] += 1


        self.confusion_matrices_default = confusion_matrices_default

        # this stores a matrix which corresponds to the uncovered labels of all examples, where uncovered labels are
        # represented by a one and covered examples are represented by a zero
        uncovered_labels[:,:] = 1
        self.uncovered_labels = uncovered_labels

        self.minority_labels = minority_labels
        self.true_labels = y

        return default_rule

    cdef begin_instance_sub_sampling(self):
        pass

    cdef update_sub_sample(self, intp example_index):
        pass

    cdef begin_search(self, intp[::1] label_indices):
        cdef uint32[::1] minority_labels
        cdef intp num_labels

        if label_indices is None:
            minority_labels = self.minority_labels
            num_labels = minority_labels.shape[0]
        else:
            num_labels = label_indices.shape[0]

        cdef LabelIndependentPrediction prediction = self.prediction
        cdef float64[::1] predicted_scores
        cdef float64[::1] quality_scores
        cdef float64[::1, :] confusion_matrices_covered = self.confusion_matrices_covered

        if confusion_matrices_covered is None or confusion_matrices_covered.shape[0] != num_labels:
                    confusion_matrices_covered = matrix_float64(num_labels, 4)
                    self.confusion_matrices_covered = confusion_matrices_covered
                    predicted_scores = array_float64(num_labels)
                    prediction.predicted_scores = predicted_scores
                    quality_scores = array_float64(num_labels)
                    prediction.quality_scores = quality_scores

        confusion_matrices_covered[:,:] = 0
        self.label_indices = label_indices

    cdef update_search(self, intp example_index, uint32 weight):
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint32[::1] minority_labels = self.minority_labels
        cdef uint8[::1, :] true_labels = self.true_labels
        cdef float64[::1, :] confusion_matrices_covered = self.confusion_matrices_covered
        cdef intp[::1] label_indices = self.label_indices
        cdef intp num_labels = true_labels.shape[1]
        cdef intp c, l
        cdef uint8 true_label, predicted_label

        for c in range(num_labels):
            l = get_index(c, label_indices)
            if uncovered_labels[example_index, l] > 0:
                true_label = true_labels[example_index, c]
                predicted_label = minority_labels[c]

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

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        cdef LabelIndependentPrediction prediction = self.prediction
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores
        cdef float64 overall_quality_score = 0
        cdef uint32[::1] minority_labels = self.minority_labels
        cdef intp num_labels = minority_labels.shape[0]
        cdef float64[::1, :] confusion_matrices_covered = self.confusion_matrices_covered
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        cdef intp c

        if predicted_scores is None or predicted_scores.shape[0] != num_labels:
            predicted_scores = array_float64(num_labels)
            prediction.predicted_scores = predicted_scores
            quality_scores = array_float64(num_labels)
            prediction.quality_scores = quality_scores

        for c in range(num_labels):
            predicted_scores[c] = minority_labels[c]
            if uncovered:
                quality_scores[c] = self.evaluate_confusion_matrix(
                    confusion_matrices_default[c, _IN] - confusion_matrices_covered[c, _IN],
                    confusion_matrices_default[c, _IP] - confusion_matrices_covered[c, _IP],
                    confusion_matrices_default[c, _RN] - confusion_matrices_covered[c, _RN],
                    confusion_matrices_default[c, _RP] - confusion_matrices_covered[c, _RP],
                    confusion_matrices_covered[c, _IN],
                    confusion_matrices_covered[c, _IP],
                    confusion_matrices_covered[c, _RN],
                    confusion_matrices_covered[c, _RP]
                )
            else:
                quality_scores[c] = self.evaluate_confusion_matrix(
                    confusion_matrices_covered[c, _IN],
                    confusion_matrices_covered[c, _IP],
                    confusion_matrices_covered[c, _RN],
                    confusion_matrices_covered[c, _RP],
                    confusion_matrices_default[c, _IN] - confusion_matrices_covered[c, _IN],
                    confusion_matrices_default[c, _IP] - confusion_matrices_covered[c, _IP],
                    confusion_matrices_default[c, _RN] - confusion_matrices_covered[c, _RN],
                    confusion_matrices_default[c, _RP] - confusion_matrices_covered[c, _RP],
                )

            overall_quality_score += quality_scores[c]

        prediction.overall_quality_score = overall_quality_score / num_labels

        return prediction

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        return self.evaluate_label_independent_predictions(uncovered)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint8[::1, :] true_labels = self.true_labels
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        cdef uint32[::1] minority_labels = self.minority_labels
        cdef intp num_labels = predicted_scores.shape[0]
        cdef float64 minority_label
        cdef intp l, i

        # Only the labels that are predicted by the new rule must be considered
        for l in label_indices:
            minority_label = minority_labels[l]

            # Only the examples that are covered by the new rule must be considered
            for i in covered_example_indices:
                uncovered_labels[i, l] = 0

                # Remove covered labels from the confusion matrices of the default rule
                true_label = true_labels[i, l]

                if true_label == 0:
                    if minority_label == 1:  # i.e., default rule predicts 0
                        confusion_matrices_default[l, _IN] -= 1
                    elif minority_label == 0:  # i.e., default rule predicts 1
                        confusion_matrices_default[l, _IP] -= 1
                elif true_label == 1:
                    if minority_label == 1:  # i.e., default rule predicts 0
                        confusion_matrices_default[l, _RN] -= 1
                    elif minority_label == 0:  # i.e., default rule predicts 1
                        confusion_matrices_default[l, _RP] -= 1

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        # TODO extract logic to new heuristic class
        return (cip + crn + uip + urn) / (cin + cip + crn + crp + uin + uip + urn + urp)