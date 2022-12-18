from mlrl.common.cython._types cimport uint32

from libcpp.memory cimport unique_ptr


cdef extern from "common/input/feature_info.hpp" nogil:

    cpdef enum FeatureTypeImpl"IFeatureInfo::FeatureType":

        BINARY"IFeatureInfo::FeatureType::BINARY" = 0

        NOMINAL"IFeatureInfo::FeatureType::NOMINAL" = 1

        NUMERICAL_OR_ORDINAL"IFeatureInfo::FeatureType::NUMERICAL_OR_ORDINAL" = 2


    cdef cppclass IFeatureInfo:
        pass


cdef extern from "common/input/feature_info_equal.hpp" nogil:

    cdef cppclass IEqualFeatureInfo(IFeatureInfo):
        pass


    unique_ptr[IEqualFeatureInfo] createEqualFeatureInfo(FeatureTypeImpl featureType)


cdef extern from "common/input/feature_info_mixed.hpp" nogil:

    cdef cppclass IMixedFeatureInfo(IFeatureInfo):

        # Functions:

        void setFeatureType(uint32 featureIndex, FeatureTypeImpl featureType)


    unique_ptr[IMixedFeatureInfo] createMixedFeatureInfo(uint32 numFeatures)


cdef class FeatureInfo:

    # Functions:

    cdef IFeatureInfo* get_feature_info_ptr(self)


cdef class EqualFeatureInfo(FeatureInfo):

    # Attributes:

    cdef unique_ptr[IEqualFeatureInfo] feature_info_ptr


cdef class MixedFeatureInfo(FeatureInfo):

    # Attributes:

    cdef unique_ptr[IMixedFeatureInfo] feature_info_ptr
