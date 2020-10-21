from boomer.common._arrays cimport float64


cdef extern from "cpp/rule_evaluation.h" nogil:

    cdef cppclass EvaluatedPrediction:

        # Attributes:

        float64 overallQualityScore
