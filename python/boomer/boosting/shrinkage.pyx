"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for shrinking the weights of rules.
"""
from boomer.common._arrays cimport intp


cdef class Shrinkage(PostProcessor):
    """
    A base class for all classes that allow to post-process the rules by shrinking their weight. The shrinkage
    parameter, a.k.a. the learning rate, may be constant or a function depending on the number of rules learned so far.
    """

    cdef void post_process(self, HeadCandidate* head):
        pass


cdef class ConstantShrinkage(Shrinkage):
    """
    Shrinks the weights of rules by a constant shrinkage parameter.
    """

    def __cinit__(self, float64 shrinkage = 1.0):
        """
        :param shrinkage: The shrinkage parameter. Must be in (0, 1]
        """
        self.shrinkage = shrinkage

    cdef void post_process(self, HeadCandidate* head):
        cdef float64 shrinkage = self.shrinkage
        cdef intp num_predictions = head.numPredictions_
        cdef float64* predicted_scores = head.predictedScores_
        cdef intp c

        for c in range(num_predictions):
            predicted_scores[c] *= shrinkage
