from boomer.common._arrays cimport float64

"""
An enum that specified all positive elements of a confusion matrix.
"""
cdef enum Element:
    IN = 0
    IP = 1
    RN = 2
    RP = 3


cdef class Heuristic:

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp)


cdef class HammingLoss(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp)


cdef class Precision(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp)


cdef class Recall(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp)


cdef class WeightedRelativeAccuracy(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp)


cdef class FMeasure(Heuristic):

    # Attributes:

    cdef readonly float64 beta

    cdef Recall recall

    cdef Precision precision

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp)


cdef class MEstimate(Heuristic):

    # Attributes:

    cdef readonly float64 m

    cdef WeightedRelativeAccuracy wra

    cdef Precision precision

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp)
