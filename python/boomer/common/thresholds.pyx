"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to the thresholds that may be used by the conditions of rules.
"""


cdef class ThresholdsFactory:
    """
    A base class for all factories that allow to create instances of the class `AbstractThresholds`.
    """

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistics_provider):
        """
        Creates and returns a new instance of the class `AbstractThresholds`.

        :param feature_matrix:          A `FeatureMatrix` that provides access to the feature values of the training
                                        examples
        :param nominal_feature_vector:  A `NominalFeatureVector` that provides access to the information whether
                                        individual features are nominal or not
        :param statistics_provider:     A `StatisticsProvider` that provides access to statistics about the labels of
                                        training examples
        :return:                        A pointer to an object of type `AbstractThresholds` that has been created
        """
        pass


cdef class ExactThresholdsFactory(ThresholdsFactory):
    """
    A factory that allows to create instances of the class `ExactThresholdsImpl`.
    """

    cdef AbstractThresholds* create(self, FeatureMatrix feature_matrix, NominalFeatureVector nominal_feature_vector,
                                    StatisticsProvider statistics_provider):
        return new ExactThresholdsImpl(feature_matrix.feature_matrix_ptr,
                                       nominal_feature_vector.nominal_feature_vector_ptr,
                                       statistics_provider.statistics_ptr)
