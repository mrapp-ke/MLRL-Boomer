from boomer.common._arrays cimport float64

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/heuristics.h" namespace "seco" nogil:

    cdef cppclass AbstractHeuristic:

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp)


    cdef cppclass PrecisionImpl(AbstractHeuristic):

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp)


    cdef cppclass RecallImpl(AbstractHeuristic):

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp)


    cdef cppclass WRAImpl(AbstractHeuristic):

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp)


    cdef cppclass HammingLossImpl(AbstractHeuristic):

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp)


    cdef cppclass FMeasureImpl(AbstractHeuristic):

        # Constructors:

        FMeasureImpl(float64 beta) except +

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp)


    cdef cppclass MEstimateImpl(AbstractHeuristic):

        # Constructors:

        MEstimateImpl(float64 m) except +

        # Functions:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp)


cdef class Heuristic:

    # Attributes:

    cdef shared_ptr[AbstractHeuristic] heuristic_ptr


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
