from boomer.common._types cimport uint32
from boomer.common.head_refinement cimport HeadRefinementFactory, AbstractPrediction
from boomer.common.input_data cimport FeatureMatrix, NominalFeatureMask
from boomer.common.rule_refinement cimport Refinement
from boomer.common.statistics cimport StatisticsProvider
from boomer.common.sampling cimport IWeightVector

from libcpp.memory cimport unique_ptr


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


cdef class ThresholdsFactory:

    # Functions:

    cdef IThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureMask nominal_feature_mask,
                             StatisticsProvider statistic_provider, HeadRefinementFactory head_refinement_factory)
