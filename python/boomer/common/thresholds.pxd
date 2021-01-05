from boomer.common._types cimport uint32
from boomer.common.head_refinement cimport IHeadRefinementFactory, AbstractPrediction
from boomer.common.input cimport IFeatureMatrix, INominalFeatureMask
from boomer.common.rule_refinement cimport Refinement
from boomer.common.statistics cimport IStatistics
from boomer.common.sampling cimport IWeightVector

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/thresholds/coverage_mask.h" nogil:

    cdef cppclass CoverageMask:
        pass


cdef extern from "cpp/thresholds/thresholds_subset.h" nogil:

    cdef cppclass IThresholdsSubset:

        # Functions:

        void filterThresholds(Refinement &refinement)

        const CoverageMask& getCoverageMask()

        void recalculatePrediction(const CoverageMask& coverageMask, Refinement &refinement)

        void applyPrediction(AbstractPrediction& prediction)


cdef extern from "cpp/thresholds/thresholds.h" nogil:

    cdef cppclass IThresholds:

        # Functions:

        unique_ptr[IThresholdsSubset] createSubset(const IWeightVector& weights)

        uint32 getNumExamples()

        uint32 getNumFeatures()

        uint32 getNumLabels()


cdef extern from "cpp/thresholds/thresholds_factory.h" nogil:

    cdef cppclass IThresholdsFactory:

        # Functions:

        unique_ptr[IThresholds] create(shared_ptr[IFeatureMatrix] featureMatrixPtr,
                                       shared_ptr[INominalFeatureMask] nominalFeatureMaskPtr,
                                       shared_ptr[IStatistics] statisticsPtr,
                                       shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr)


cdef class ThresholdsFactory:

    # Attributes:

    cdef shared_ptr[IThresholdsFactory] thresholds_factory_ptr
