"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to the thresholds that may be used by the conditions of rules.
"""


cdef class ThresholdsFactory:
    """
    A base class for all factories that allow to create instances of the class `IThresholds`.
    """

    cdef IThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureMask nominal_feature_mask,
                             StatisticsProvider statistics_provider, HeadRefinementFactory head_refinement_factory):
        """
        Creates and returns a new instance of the class `IThresholds`.

        :param feature_matrix:          A `FeatureMatrix` that provides access to the feature values of the training
                                        examples
        :param nominal_feature_mask:    A `NominalFeatureMask` that provides access to the information whether
                                        individual features are nominal or not
        :param statistics_provider:     A `StatisticsProvider` that provides access to statistics about the labels of
                                        training examples
        :param head_refinement_factory: A `HeadRefinementFactory` that allows to create instances of the class that
                                        should be used to find the heads of rules
        :return:                        A pointer to an object of type `IThresholds` that has been created
        """
        pass
