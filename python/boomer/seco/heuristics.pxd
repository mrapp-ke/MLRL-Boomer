from boomer.common._arrays cimport float64


cdef extern from "cpp/heuristics.h" namespace "seco":

    cdef enum ConfusionMatrixElement:
        IN
        IP
        RN
        RP


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


    cdef cppclass MEstimateImpl(AbstractHeuristic):

        # Constructors:

        MEstimateImpl(float64 m)

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) nogil


cdef class Heuristic:

    # Attributes:

    cdef AbstractHeuristic* heuristic


cdef class Precision(Heuristic):
    pass


cdef class Recall(Heuristic):
    pass


cdef class WRA(Heuristic):
    pass


cdef class HammingLoss(Heuristic):
    pass


cdef class FMeasure(Heuristic):
    pass


cdef class MEstimate(Heuristic):
    pass
