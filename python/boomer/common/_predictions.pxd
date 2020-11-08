"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

classes that store the predictions of rules, as well as corresponding quality scores.
"""
from boomer.common._types cimport uint32, float64
from boomer.common.statistics cimport AbstractStatistics

from libcpp cimport bool


cdef extern from "cpp/predictions.h" nogil:

    cdef cppclass AbstractPrediction:

        ctypedef float64* score_const_iterator

        # Functions:

        bool isPartial()

        uint32 getNumElements()

        score_const_iterator scores_cbegin()

        score_const_iterator scores_cend()

        void apply(AbstractStatistics& statistics, uint32 statisticIndex)


    cdef cppclass AbstractEvaluatedPrediction(AbstractPrediction):
        pass


    cdef cppclass PartialPrediction(AbstractEvaluatedPrediction):

        ctypedef uint32* index_const_iterator

        # Functions:

        index_const_iterator indices_cbegin()

        index_const_iterator indices_cend()
