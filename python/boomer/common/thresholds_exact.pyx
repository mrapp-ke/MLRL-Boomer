"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to exact thresholds that may be used by the conditions of
rules.
"""


cdef class ExactThresholdsFactory(ThresholdsFactory):
    """
    A factory that allows to create instances of the class `ExactThresholds`.
    """

    cdef IThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureMask nominal_feature_mask,
                             StatisticsProvider statistics_provider, HeadRefinementFactory head_refinement_factory):
        return new ExactThresholds(feature_matrix.feature_matrix_ptr, nominal_feature_mask.nominal_feature_mask_ptr,
                                   statistics_provider.statistics_ptr,
                                   head_refinement_factory.head_refinement_factory_ptr)
