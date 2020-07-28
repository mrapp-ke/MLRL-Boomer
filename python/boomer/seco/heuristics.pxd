from boomer.common._arrays cimport float64


cdef extern from "cpp/heuristics.h" namespace "heuristics":

    cdef cppclass AbstractHeuristic:

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) nogil


    cdef cppclass PrecisionImpl(AbstractHeuristic):

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) nogil


    cdef cppclass RecallImpl(AbstractHeuristic):

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) nogil


    cdef cppclass WRAImpl(AbstractHeuristic):

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) nogil


    cdef cppclass HammingLossImpl(AbstractHeuristic):

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) nogil


    cdef cppclass FMeasureImpl(AbstractHeuristic):

        # Constructors:

        FMeasureImpl(float64 beta)

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) nogil


    cdef cppclass MEstimateFunction(AbstractHeuristic):

        # Constructors:

        MEstimateFunction(float64 m)

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) nogil


"""
An enum that specified all positive elements of a confusion matrix.
"""
cdef enum Element:
    IN = 0
    IP = 1
    RN = 2
    RP = 3


cdef class Heuristic:

    # Attributes:

    cdef AbstractHeuristic* heuristic

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp) nogil


cdef class Precision(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp) nogil


cdef class Recall(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp) nogil


cdef class WRA(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp) nogil


cdef class HammingLoss(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp) nogil


cdef class FMeasure(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp) nogil


cdef class MEstimate(Heuristic):

    # Functions:

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp) nogil
