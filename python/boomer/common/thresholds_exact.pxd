from boomer.common.head_refinement cimport HeadRefinementFactory, IHeadRefinementFactory
from boomer.common.input_data cimport FeatureMatrix, IFeatureMatrix, NominalFeatureMask, INominalFeatureMask
from boomer.common.statistics cimport StatisticsProvider, IStatistics
from boomer.common.thresholds cimport ThresholdsFactory, AbstractThresholds

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/thresholds_exact.h" nogil:

    cdef cppclass ExactThresholds(AbstractThresholds):

        # Constructors:

        ExactThresholds(shared_ptr[IFeatureMatrix] featureMatrixPtr,
                        shared_ptr[INominalFeatureMask] nominalFeatureMaskPtr,
                        shared_ptr[IStatistics] statisticsPtr,
                        shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr) except +


cdef class ExactThresholdsFactory(ThresholdsFactory):

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureMask nominal_feature_mask,
                                    StatisticsProvider statistic_provider,
                                    HeadRefinementFactory head_refinement_factory)
