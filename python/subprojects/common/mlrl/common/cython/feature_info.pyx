"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move

from mlrl.common.cython._types cimport uint32


cdef class FeatureInfo:
    """
    Provides information about the types of individual features.
    """

    cdef IFeatureInfo* get_feature_info_ptr(self):
        pass


cdef class EqualFeatureInfo(FeatureInfo):
    """
    Provides information about the types of individual features in cases where all features are of the same type, i.e.,
    where all features are either ordinal, nominal or numerical.
    """

    @classmethod
    def create_ordinal(cls) -> EqualFeatureInfo:
        cdef EqualFeatureInfo equal_feature_info = EqualFeatureInfo.__new__(EqualFeatureInfo)
        equal_feature_info.feature_info_ptr = createOrdinalFeatureInfo()
        return equal_feature_info

    @classmethod
    def create_nominal(cls) -> EqualFeatureInfo:
        cdef EqualFeatureInfo equal_feature_info = EqualFeatureInfo.__new__(EqualFeatureInfo)
        equal_feature_info.feature_info_ptr = createNominalFeatureInfo()
        return equal_feature_info

    @classmethod
    def create_numerical(cls) -> EqualFeatureInfo:
        cdef EqualFeatureInfo equal_feature_info = EqualFeatureInfo.__new__(EqualFeatureInfo)
        equal_feature_info.feature_info_ptr = createNumericalFeatureInfo()
        return equal_feature_info


    cdef IFeatureInfo* get_feature_info_ptr(self):
        return self.feature_info_ptr.get()


cdef class MixedFeatureInfo(FeatureInfo):
    """
    Provides information about the types of individual features in cases where different types of features, i.e.,
    ordinal, nominal and numerical ones, are available.
    """

    def __cinit__(self, uint32 num_features, const uint32[::1] ordinal_feature_indices not None,
                  const uint32[::1] nominal_feature_indices not None):
        """
        :param num_features:            The total number of available features
        :param ordinal_feature_indices: A C-contiguous array of type `uint32`, shape `(num_ordinal_features)`, that
                                        stores the indices of all ordinal features
        :param nominal_feature_indices: A C-contiguous array of type `uint32`, shape `(num_ordinal_features)`, that
                                        stores the indices of all nominal features
        """
        cdef unique_ptr[IMixedFeatureInfo] feature_info_ptr = createMixedFeatureInfo(num_features)
        cdef uint32 num_ordinal_features = ordinal_feature_indices.shape[0]
        cdef uint32 num_nominal_features = nominal_feature_indices.shape[0]
        cdef uint32 i

        for i in range(num_nominal_features):
            feature_info_ptr.get().setNominal(nominal_feature_indices[i])

        for i in range(num_ordinal_features):
            feature_info_ptr.get().setOrdinal(ordinal_feature_indices[i])

        self.feature_info_ptr = move(feature_info_ptr)

    cdef IFeatureInfo* get_feature_info_ptr(self):
        return self.feature_info_ptr.get()
