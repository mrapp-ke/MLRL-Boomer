from boomer.common._arrays cimport uint32, intp
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix
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

    # Functions:

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction)

    cdef void reset_sampled_statistics(self)

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight)

    cdef void reset_covered_statistics(self)

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove)

    cdef AbstractRefinementSearch* begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head)
