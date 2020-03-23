from boomer.algorithm._arrays cimport float64

cdef class Heuristic:
    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        pass

cdef class HammingLoss(Heuristic):
    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        return (cip + crn + uin + urp) / (cin + cip + crn + crp + uin + uip + urn + urp)
