from boomer.common._arrays cimport uint32
from boomer.common._tuples cimport IndexedFloat32Array, IndexedFloat32ArrayWrapper
from boomer.common._predictions cimport Prediction
from boomer.common.head_refinement cimport IHeadRefinement
from boomer.common.input_data cimport FeatureMatrix, IFeatureMatrix, NominalFeatureVector, INominalFeatureVector
from boomer.common.rule_refinement cimport IRuleRefinement, Refinement
from boomer.common.statistics cimport StatisticsProvider, AbstractStatistics
from boomer.common.sub_sampling cimport IWeightVector

from libcpp.memory cimport shared_ptr
from libcpp.unordered_map cimport unordered_map


cdef extern from "cpp/thresholds.h" nogil:

    cdef cppclass IThresholdsSubset:

        # Functions:

        IRuleRefinement* createRuleRefinement(uint32 featureIndex)

        uint32 applyRefinement(Refinement &refinement)

        Prediction* calculateOverallPrediction(IHeadRefinement* headRefinement, uint32 numLabelIndices,
                                               const uint32* labelIndices)

        void applyPrediction(Prediction* prediction)

    cdef cppclass AbstractThresholds:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()

        uint32 getNumLabels()

        IThresholdsSubset* createSubset(IWeightVector* weights)


    cdef cppclass ExactThresholdsImpl(AbstractThresholds):

        # Constructors:

        ExactThresholdsImpl(shared_ptr[IFeatureMatrix] featureMatrixPtr,
                            shared_ptr[INominalFeatureVector] nominalFeatureVectorPtr,
                            shared_ptr[AbstractStatistics] statisticsPtr) except +

        # Attributes:

        unordered_map[uint32, IndexedFloat32Array*] cache_


cdef extern from "cpp/thresholds.h" namespace "ExactThresholdsImpl" nogil:

    cdef cppclass ThresholdsSubsetImpl(IThresholdsSubset):

        # Attributes:

        unordered_map[uint32, IndexedFloat32ArrayWrapper*] cacheFiltered_


cdef class ThresholdsFactory:

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistic_provider)


cdef class ExactThresholdsFactory(ThresholdsFactory):

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistic_provider)
