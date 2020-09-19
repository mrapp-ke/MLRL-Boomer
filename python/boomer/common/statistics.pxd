from boomer.common._arrays cimport uint32
from boomer.common._data cimport AbstractMatrix
from boomer.common._predictions cimport Prediction, PredictionCandidate, LabelWisePredictionCandidate
from boomer.common.input_data cimport LabelMatrix

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/statistics.h" nogil:

    cdef cppclass AbstractStatisticsSubset:

        # Functions:

        void addToSubset(uint32 statisticIndex, uint32 weight)

        void resetSubset()

        LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered, bool accumulated) except +

        PredictionCandidate* calculateExampleWisePrediction(bool uncovered, bool accumulated) except +


    cdef cppclass AbstractDecomposableStatisticsSubset(AbstractStatisticsSubset):
        pass


    cdef cppclass AbstractStatistics(AbstractMatrix):

        # Functions:

        void resetSampledStatistics()

        void addSampledStatistic(uint32 statisticIndex, uint32 weight)

        void resetCoveredStatistics()

        void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove)

        AbstractStatisticsSubset* createSubset(uint32 numLabelIndices, const uint32* labelIndices)

        void applyPrediction(uint32 statisticIndex, Prediction* prediction)


cdef class StatisticsProvider:

    # Functions:

    cdef AbstractStatistics* get(self)

    cdef void switch_rule_evaluation(self)


cdef class StatisticsProviderFactory:

    # Functions:

    cdef StatisticsProvider create(self, LabelMatrix label_matrix)
