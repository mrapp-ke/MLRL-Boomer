from boomer.common.binning cimport Binning, IBinning
from boomer.common.head_refinement cimport HeadRefinementFactory, IHeadRefinementFactory
from boomer.common.input_data cimport FeatureMatrix, IFeatureMatrix, NominalFeatureVector, INominalFeatureVector
from boomer.common.statistics cimport StatisticsProvider, AbstractStatistics
from boomer.common.thresholds cimport ThresholdsFactory, AbstractThresholds

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/thresholds_approximate.h" nogil:

    cdef cppclass ApproximateThresholds(AbstractThresholds):

        # Constructors:

        ApproximateThresholds(shared_ptr[IFeatureMatrix] featureMatrixPtr,
                              shared_ptr[INominalFeatureVector] nominalFeatureVectorPtr,
                              shared_ptr[AbstractStatistics] statisticsPtr,
                              shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr,
                              shared_ptr[IBinning] binningPtr) except +


cdef class ApproximateThresholdsFactory(ThresholdsFactory):

    # Attributes:

    cdef Binning binning

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistic_provider,
                                    HeadRefinementFactory head_refinement_factory)