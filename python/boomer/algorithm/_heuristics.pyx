cdef class Heuristic:
    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        pass

cdef class HammingLoss(Heuristic):
    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 num_labels = cin + cip + crn + crp + uin + uip + urn + urp
        if num_labels == 0:
            return 1
        return (cip + crn + urn + urp) / num_labels

cdef class Precision(Heuristic):
    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 num_covered_labels = cin + cip + crn + crp
        if num_covered_labels == 0:
            return 1
        return (cip + crn) / num_covered_labels
