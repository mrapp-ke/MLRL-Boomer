from boomer.common._arrays cimport uint32, intp
from boomer.common._predictions cimport Prediction, PredictionCandidate, LabelWisePredictionCandidate
from boomer.common.input_data cimport LabelMatrix

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/statistics.h" nogil:

    cdef cppclass AbstractRefinementSearch:

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight)

        void resetSearch() nogil

        LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered, bool accumulated) except +

        PredictionCandidate* calculateExampleWisePrediction(bool uncovered, bool accumulated) except +


    cdef cppclass AbstractDecomposableRefinementSearch(AbstractRefinementSearch):

        # Functions:

        void updateSearch(intp statisticIndex, uint32 weight)

        void resetSearch()

        LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered, bool accumulated) except +

        PredictionCandidate* calculateExampleWisePrediction(bool uncovered, bool accumulated) except +


    cdef cppclass AbstractStatistics:

        # Attributes:

        intp numStatistics_

        # Functions:

        void resetSampledStatistics()

        void addSampledStatistic(intp statisticIndex, uint32 weight)

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, Prediction* prediction)


    cdef cppclass AbstractStatisticsFactory:

        # Functions:

        AbstractStatistics* create()


cdef class StatisticsFactory:

    # Attributes:

    cdef shared_ptr[AbstractStatisticsFactory] statistics_factory_ptr

    # Functions:

    cdef AbstractStatistics* create_initial_statistics(self, LabelMatrix label_matrix)

    cdef AbstractStatistics* copy_statistics(self, AbstractStatistics* statistics)
