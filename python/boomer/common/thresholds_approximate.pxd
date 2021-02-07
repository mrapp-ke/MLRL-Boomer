from boomer.common.binning cimport IFeatureBinning
from boomer.common.thresholds cimport ThresholdsFactory, IThresholdsFactory

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/thresholds/thresholds_approximate.hpp" nogil:

    cdef cppclass ApproximateThresholdsFactoryImpl"ApproximateThresholdsFactory"(IThresholdsFactory):

        # Constructors:

        ApproximateThresholdsFactory(shared_ptr[IFeatureBinning] binningPtr) except +


cdef class ApproximateThresholdsFactory(ThresholdsFactory):
    pass
