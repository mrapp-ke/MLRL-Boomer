from boomer.common._arrays cimport uint32, intp
from boomer.common._predictions cimport Prediction
from boomer.common.input_data cimport AbstractLabelMatrix
from boomer.common.statistics cimport Statistics, AbstractStatistics, AbstractRefinementSearch

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/statistics.h" namespace "boosting" nogil:

    cdef cppclass AbstractGradientStatistics(AbstractStatistics):

        # Functions:

        void applyDefaultPrediction(shared_ptr[AbstractLabelMatrix] labelMatrixPtr, Prediction* defaultPrediction)

        void resetSampledStatistics()

        void addSampledStatistic(intp statisticIndex, uint32 weight)

        void resetCoveredStatistics()

        void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove)

        AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices)

        void applyPrediction(intp statisticIndex, const intp* labelIndices, Prediction* prediction)


cdef class GradientStatistics(Statistics):
    pass
