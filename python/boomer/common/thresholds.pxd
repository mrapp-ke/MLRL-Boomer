from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport AbstractPrediction
from boomer.common.head_refinement cimport HeadRefinementFactory
from boomer.common.input_data cimport FeatureMatrix, NominalFeatureVector
from boomer.common.rule_refinement cimport Refinement
from boomer.common.statistics cimport StatisticsProvider
from boomer.common.sub_sampling cimport IWeightVector

from libcpp.memory cimport unique_ptr


cdef extern from "cpp/thresholds.h" nogil:

    cdef cppclass CoverageMask:
        pass


    cdef cppclass IThresholdsSubset:

        # Functions:

        void applyRefinement(Refinement &refinement)

        const CoverageMask& getCoverageMask()

        void recalculatePrediction(const CoverageMask& coverageMask, Refinement &refinement)

        void applyPrediction(AbstractPrediction& prediction)


    cdef cppclass AbstractThresholds:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()

        uint32 getNumLabels()

        unique_ptr[IThresholdsSubset] createSubset(const IWeightVector& weights)


cdef class ThresholdsFactory:

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistic_provider,
                                    HeadRefinementFactory head_refinement_factory)
