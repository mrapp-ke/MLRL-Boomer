from boomer.common._types cimport uint32
from boomer.common.head_refinement cimport IHeadRefinementFactory
from boomer.common.input cimport IFeatureMatrix, INominalFeatureMask
from boomer.common.statistics cimport IStatisticsProvider

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/thresholds/thresholds.h" nogil:

    cdef cppclass IThresholds:

        # Functions:

        uint32 getNumExamples()

        uint32 getNumLabels()


cdef extern from "cpp/thresholds/thresholds_factory.h" nogil:

    cdef cppclass IThresholdsFactory:

        # Functions:

        unique_ptr[IThresholds] create(shared_ptr[IFeatureMatrix] featureMatrixPtr,
                                       shared_ptr[INominalFeatureMask] nominalFeatureMaskPtr,
                                       shared_ptr[IStatisticsProvider] statisticsProviderPtr,
                                       shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr)


cdef class ThresholdsFactory:

    # Attributes:

    cdef shared_ptr[IThresholdsFactory] thresholds_factory_ptr
