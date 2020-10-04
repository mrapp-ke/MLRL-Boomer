from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport Prediction
from boomer.common.head_refinement cimport IHeadRefinement
from boomer.common.input_data cimport FeatureMatrix, IFeatureMatrix, NominalFeatureVector, INominalFeatureVector
from boomer.common.rule_refinement cimport AbstractRuleRefinement, Refinement
from boomer.common.statistics cimport StatisticsProvider, AbstractStatistics
from boomer.common.sub_sampling cimport IWeightVector

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/thresholds.h" nogil:

    cdef cppclass IThresholdsSubset:

        # Functions:

        AbstractRuleRefinement* createRuleRefinement(uint32 featureIndex)

        void applyRefinement(Refinement &refinement)

        void recalculatePrediction(IHeadRefinement& headRefinement, Refinement &refinement)

        void applyPrediction(Prediction& prediction)


    cdef cppclass AbstractThresholds:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()

        uint32 getNumLabels()

        unique_ptr[IThresholdsSubset] createSubset(shared_ptr[IWeightVector] weightsPtr)


    cdef cppclass ExactThresholdsImpl(AbstractThresholds):

        # Constructors:

        ExactThresholdsImpl(shared_ptr[IFeatureMatrix] featureMatrixPtr,
                            shared_ptr[INominalFeatureVector] nominalFeatureVectorPtr,
                            shared_ptr[AbstractStatistics] statisticsPtr) except +


cdef class ThresholdsFactory:

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistic_provider)


cdef class ExactThresholdsFactory(ThresholdsFactory):

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistic_provider)
