from boomer.common._arrays cimport array_float64, fortran_matrix_float64, array_uint8, get_index


DEF _IN = 0
DEF _IP = 1
DEF _RN = 2
DEF _RP = 3


cdef class LabelWiseRefinementSearch(DecomposableRefinementSearch):
    """
    Allows to search for the best refinement of a rule according to a coverage loss that uses label-wise averaging.
    """

    def __cinit__(self, Heuristic heuristic, intp[::1] label_indices, const uint8[::1, :] true_labels,
                  const float64[::1, :] uncovered_labels, const uint8[::1] minority_labels,
                  const float64[::1, :] confusion_matrices_default):
        """
        :param heuristic:                   The heuristic to be used
        :param label_indices:               An array of dtype int, shape `(num_considered_labels)`, representing the
                                            indices of the labels that should be considered by the search or None, if
                                            all labels should be considered
        :param true_labels:                 An array of dtype uint, shape `(num_examples, num_labels)`, representing the
                                            true labels according to the ground truth
        :param uncovered_labels:            An array of dtype float, shape `(num_examples, num_labels)`, indicating
                                            which each examples and labels remain to be covered
        :param minority_labels:             An array of dtype uint, shape `(num_labels)`, representing the minority
                                            class for each label
        :param confusion_matrices_default:  An array of dtype float, shape `(4, num_labels)`, representing a confusion
                                            matrix that stores the elements of all examples per label
        """
        self.heuristic = heuristic
        self.label_indices = label_indices
        self.true_labels = true_labels
        self.uncovered_labels = uncovered_labels
        self.minority_labels = minority_labels
        self.confusion_matrices_default = confusion_matrices_default
        self.accumulated_confusion_matrices_covered = None
        cdef LabelWisePrediction prediction = LabelWisePrediction.__new__(LabelWisePrediction)
        cdef intp num_labels = minority_labels.shape[0] if label_indices is None else label_indices.shape[0]
        cdef float64[::1, :] confusion_matrices_covered = fortran_matrix_float64(num_labels, 4)
        confusion_matrices_covered[:, :] = 0
        self.confusion_matrices_covered = confusion_matrices_covered
        self.accumulated_confusion_matrices_covered = None
        cdef float64[::1] predicted_scores = array_float64(num_labels)
        prediction.predicted_scores = predicted_scores
        cdef float64[::1] quality_scores = array_float64(num_labels)
        prediction.quality_scores = quality_scores
        self.prediction = prediction

    cdef void update_search(self, intp example_index, uint32 weight):
        cdef const float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef const uint8[::1] minority_labels = self.minority_labels
        cdef const uint8[::1, :] true_labels = self.true_labels
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

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        cdef LabelWisePrediction prediction = self.prediction
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores
        cdef float64 overall_quality_score = 0
        cdef const uint8[::1] minority_labels = self.minority_labels
        cdef intp[::1] label_indices = self.label_indices
        cdef float64[::1, :] confusion_matrices_covered = self.accumulated_confusion_matrices_covered if accumulated else self.confusion_matrices_covered
        cdef const float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
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


cdef class LabelWiseAveraging(CoverageLoss):
    """
    Allows to locally minimize a coverage loss that uses label-wise averaging by the rules that are learned by an
    algorithm based on sequential covering.
    """

    def __cinit__(self, Heuristic heuristic):
        """
        :param heuristic: The heuristic to be used
        """
        self.heuristic = heuristic

    cdef DefaultPrediction calculate_default_prediction(self, uint8[::1, :] y):
        cdef intp num_examples = y.shape[0]
        cdef intp num_labels = y.shape[1]
        cdef float64[::1] default_rule = array_float64(num_labels)
        cdef DefaultPrediction prediction = DefaultPrediction.__new__(DefaultPrediction)
        prediction.predicted_scores = default_rule
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

        return prediction

    cdef void begin_instance_sub_sampling(self):
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        confusion_matrices_default[:, :] = 0

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove):
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint8[::1, :] true_labels = self.true_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef intp num_labels = minority_labels.shape[0]
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        cdef float64 signed_weight = -<float64>weight if remove else weight
        cdef intp c
        cdef uint8 true_label, predicted_label

        for c in range(num_labels):
            if uncovered_labels[example_index, c] > 0:
                true_label = true_labels[example_index, c]
                predicted_label = minority_labels[c]

                if true_label == 0:
                    if predicted_label == 0:
                        confusion_matrices_default[c, _IN] += signed_weight
                    elif predicted_label == 1:
                        confusion_matrices_default[c, _IP] += signed_weight
                elif true_label == 1:
                    if predicted_label == 0:
                        confusion_matrices_default[c, _RN] += signed_weight
                    elif predicted_label == 1:
                        confusion_matrices_default[c, _RP] += signed_weight

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        cdef Heuristic heuristic = self.heuristic
        cdef uint8[::1, :] true_labels = self.true_labels
        cdef float64[::1, :] uncovered_labels = self.uncovered_labels
        cdef uint8[::1] minority_labels = self.minority_labels
        cdef float64[::1, :] confusion_matrices_default = self.confusion_matrices_default
        return LabelWiseRefinementSearch.__new__(LabelWiseRefinementSearch, heuristic, label_indices, true_labels,
                                                 uncovered_labels, minority_labels, confusion_matrices_default)

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
