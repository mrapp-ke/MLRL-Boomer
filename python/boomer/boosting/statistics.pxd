from boomer.common._arrays cimport uint32, intp
from boomer.common.input_data cimport AbstractLabelMatrix
from boomer.common.statistics cimport Statistics, AbstractStatistics, AbstractRefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction

from libcpp cimport bool


cdef extern from "cpp/statistics.h" namespace "boosting" nogil:

    cdef cppclass AbstractGradientStatistics(AbstractStatistics):

        # Functions:

        void applyDefaultPrediction(AbstractLabelMatrix* labelMatrix, DefaultPrediction* defaultPrediction)

        void resetSampledStatistics()

        void addSampledStatistic(intp statisticIndex, uint32 weight)

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head)


cdef class GradientStatistics(Statistics):
    pass
