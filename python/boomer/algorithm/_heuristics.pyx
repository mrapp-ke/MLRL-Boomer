from boomer.algorithm._arrays cimport float64

cdef class Heuristic:
    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        pass

cdef class HammingLoss(Heuristic):
    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        if cin + cip + crn + crp + uin + uip + urn + urp == 0:
            return 1
        return (cip + crn + urn + urp) / (cin + cip + crn + crp + uin + uip + urn + urp)

cdef class Precision(Heuristic):
    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        if cin + cip + crn + crp == 0:
            return 1
        return (cip + crn) / (cin + cip + crn + crp)
