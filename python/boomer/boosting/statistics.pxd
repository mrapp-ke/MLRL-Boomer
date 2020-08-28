from boomer.common._arrays cimport uint32, intp
from boomer.common._predictions cimport Prediction
from boomer.common.statistics cimport AbstractStatistics, AbstractRefinementSearch

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/statistics.h" namespace "boosting" nogil:

    cdef cppclass AbstractGradientStatistics(AbstractStatistics):

        # Functions:

        void resetSampledStatistics()

        void addSampledStatistic(intp statisticIndex, uint32 weight)

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, const intp* labelIndices, Prediction* prediction)
