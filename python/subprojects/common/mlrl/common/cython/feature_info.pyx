"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._types cimport uint8, uint32

from libcpp.utility cimport move

from enum import Enum


class FeatureType(Enum):
    """
    Specifies all supported types of features.
    """
    BINARY = 0
    NOMINAL = 1
    NUMERICAL_OR_ORDINAL = 2


cdef class FeatureInfo:
    """
    Provides information about the types of individual features.
    """

    cdef IFeatureInfo* get_feature_info_ptr(self):
        pass


cdef class EqualFeatureInfo(FeatureInfo):
    """
    Provides information about the types of individual features in cases where all features are of the same type, i.e.,
    where all features are either binary, nominal or numerical/ordinal.
    """

    def __cinit__(self, feature_type: FeatureType):
        """
        :param feature_type: A value of the enum `FeatureType` that specifies the type of all features
        """
        if feature_type == FeatureType.BINARY:
            self.feature_info_ptr = createBinaryFeatureInfo()
        elif feature_type == FeatureType.NOMINAL:
            self.feature_info_ptr = createNominalFeatureInfo()
        else:
            self.feature_info_ptr = createNumericalFeatureInfo()


    cdef IFeatureInfo* get_feature_info_ptr(self):
        return self.feature_info_ptr.get()


cdef class MixedFeatureInfo(FeatureInfo):
    """
    Provides information about the types of individual features in cases where nominal or not in cases where different
    types of features, i.e., binary, nominal and numerical/ordinal ones, are available.
    """

    def __cinit__(self, uint32 num_features, list binary_feature_indices not None,
                  list nominal_feature_indices not None):
        """
        :param num_features:            The total number of available features
        :param binary_feature_indices:  A list which contains the indices of all binary features
        :param nominal_feature_indices: A list which contains the indices of all nominal features
        """
        cdef unique_ptr[IMixedFeatureInfo] feature_info_ptr = createMixedFeatureInfo(num_features)
        cdef uint32 feature_index

        for feature_index in nominal_feature_indices:
            feature_info_ptr.get().setNominal(feature_index)

        for feature_index in binary_feature_indices:
            feature_info_ptr.get().setBinary(feature_index)

        self.feature_info_ptr = move(feature_info_ptr)

    cdef IFeatureInfo* get_feature_info_ptr(self):
        return self.feature_info_ptr.get()
