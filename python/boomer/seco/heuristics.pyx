"""
Implements different heuristics for assessing the quality of single- or multi-label rules based on confusion matrices.
Given the elements of a confusion matrix, a heuristic calculates a quality score in [0, 1].

All heuristics must be implemented as loss functions, i.e., rules with a smaller quality score are better than those
with a large quality score.

All heuristics must treat positive and negative labels equally, i.e., if the ground truth and a rule's predictions would
be inverted, the resulting quality scores must be the same as before.
"""
from libc.math cimport isinf, pow


cdef class Heuristic:
    """
    A wrapper for the abstract C++ class `HeuristicFunction`.
    """

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp) nogil:
        cdef HeuristicFunction* heuristic_function = self.heuristic_function
        cdef float64 quality_score = heuristic_function.evaluateConfusionMatrix(cin, cip, crn, crp, uin, uip, urn, urp)
        return quality_score


cdef class Precision(Heuristic):
    """
    A wrapper for the C++ class `PrecisionFunction`.
    """

    def __cinit__(self):
        self.heuristic_function = new PrecisionFunction()

    def __dealloc(self):
        del self.heuristic_function


cdef class Recall(Heuristic):
    """
    A wrapper for the C++ class `RecallFunction`.
    """

    def __cinit__(self):
        self.heuristic_function = new RecallFunction()

    def __dealloc(self):
        del self.heuristic_function


cdef class WeightedRelativeAccuracy(Heuristic):
    """
    A wrapper for the C++ class `WRAFunction`.
    """

    def __cinit__(self):
        self.heuristic_function = new WRAFunction()

    def __dealloc(self):
        del self.heuristic_function


cdef class HammingLoss(Heuristic):
    """
    A wrapper for the C++ class `HammingLossFunction`.
    """

    def __cinit__(self):
        self.heuristic_function = new HammingLossFunction()

    def __dealloc(self):
        del self.heuristic_function


cdef class FMeasure(Heuristic):
    """
    A wrapper for the C++ class `FMeasureFunction`.
    """

    def __cinit__(self, float64 beta = 0.5):
        """
        :param beta: The value of the beta-parameter. Must be at least 0
        """
        self.heuristic_function = new FMeasureFunction(beta)

    def __dealloc(self):
        del self.heuristic_function


cdef class MEstimate(Heuristic):
    """
    A wrapper for the C++ class `MEstimateFunction`.
    """

    def __cinit__(self, float64 m = 22.466):
        """
        :param m: The value of the m-parameter. Must be at least 0
        """
        self.heuristic_function = new MEstimateFunction(m)

    def __dealloc__(self):
        del self.heuristic_function
