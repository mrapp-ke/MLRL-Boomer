cdef class UncoveredLabelsCriterion(StoppingCriterion):
    """
    A stopping criterion that stops when the sum of the weight matrix used by a `CoverageLoss` is smaller than or equal
    to a certain threshold.
    """

    def __cinit__(self, CoverageLoss loss, float64 threshold):
        """
        :param loss:        The `CoverageLoss`
        :param threshold:   The threshold
        """
        self.loss = loss
        self.threshold = threshold

    cdef bint should_continue(self, intp num_rules):
        cdef CoverageLoss loss = self.loss
        cdef float64 sum_uncovered_labels = loss.sum_uncovered_labels
        cdef float64 threshold = self.threshold
        return sum_uncovered_labels > threshold
