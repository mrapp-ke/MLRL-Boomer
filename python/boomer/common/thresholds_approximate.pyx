"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to approximate thresholds that may be used by the conditions of
rules.
"""


cdef class ApproximateThresholdsFactory(ThresholdsFactory):
    """
    A factory that allows to create instances of the class `ApproximateThresholds`.
    """

    def __cinit__(self, FeatureBinning binning):
        self.binning = binning

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureMask nominal_feature_mask,
                                    StatisticsProvider statistics_provider,
                                    HeadRefinementFactory head_refinement_factory):
        return new ApproximateThresholds(feature_matrix.feature_matrix_ptr,
                                         nominal_feature_mask.nominal_feature_mask_ptr,
                                         statistics_provider.statistics_ptr,
                                         head_refinement_factory.head_refinement_factory_ptr, self.binning.binning_ptr)

