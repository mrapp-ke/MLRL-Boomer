from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport AbstractPrediction
from boomer.common.head_refinement cimport HeadRefinementFactory, IHeadRefinementFactory
from boomer.common.input_data cimport FeatureMatrix, IFeatureMatrix, NominalFeatureVector, INominalFeatureVector
from boomer.common.rule_refinement cimport Refinement
from boomer.common.statistics cimport StatisticsProvider, AbstractStatistics
from boomer.common.sub_sampling cimport IWeightVector
from boomer.common.binning cimport IBinning

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/thresholds.h" nogil:

    cdef cppclass IThresholdsSubset:

        # Functions:

        void applyRefinement(Refinement &refinement)

        void recalculatePrediction(Refinement &refinement)

        void applyPrediction(AbstractPrediction& prediction)


    cdef cppclass AbstractThresholds:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()

        uint32 getNumLabels()

        unique_ptr[IThresholdsSubset] createSubset(const IWeightVector& weights)


    cdef cppclass ExactThresholdsImpl(AbstractThresholds):

        # Constructors:

        ExactThresholdsImpl(shared_ptr[IFeatureMatrix] featureMatrixPtr,
                            shared_ptr[INominalFeatureVector] nominalFeatureVectorPtr,
                            shared_ptr[AbstractStatistics] statisticsPtr,
                            shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr) except +

    cdef cppclass ApproximateThresholdsImpl(AbstractThresholds):

        # Constructors:

        ApproximateThresholdsImpl(shared_ptr[IFeatureMatrix] featureMatrixPtr,
                                  shared_ptr[INominalFeatureVector] nominalFeatureVectorPtr,
                                  shared_ptr[AbstractStatistics] statisticsPtr,
                                  shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr,
                                  shared_ptr[IBinning] binningPtr,
                                  uint32 numBins) except +


cdef class ThresholdsFactory:

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistic_provider,
                                    HeadRefinementFactory head_refinement_factory)


cdef class ExactThresholdsFactory(ThresholdsFactory):

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistic_provider,
                                    HeadRefinementFactory head_refinement_factory)

cdef class ApproximateThresholdsFactory(ThresholdsFactory):

    # Attributes:

    cdef shared_ptr[IBinning] binningPtr

    cdef uint32 numBins

    # Constructors:

    cdef ApproximateThresholdsFactory(shared_ptr[IBinning] binning_method, uint32 num_bins)

    # Functions:

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistic_provider,
                                    HeadRefinementFactory head_refinement_factory)