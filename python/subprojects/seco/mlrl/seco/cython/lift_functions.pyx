"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class LiftFunctionFactory:
    """
    A wrapper for the pure virtual C++ class `ILiftFunctionFactory`.
    """
    pass


cdef class PeakLiftFunctionFactory(LiftFunctionFactory):
    """
    A wrapper for the C++ class `PeakLiftFunctionFactory`.
    """

    def __cinit__(self, uint32 num_labels, uint32 peak_label, float64 max_lift, float64 curvature):
        """
        :param num_labels:  The total number of available labels. Must be greater than 0
        :param peak_label:  The number of labels for which the lift is maximum. Must be in [1, numLabels]
        :param max_lift:    The lift at the peak label. Must be at least 1
        :param curvature:   The curvature of the lift function. A greater value results in a steeper curvature, a
                            smaller value results in a flatter curvature. Must be greater than 0
        """
        self.lift_function_factory_ptr = <unique_ptr[ILiftFunctionFactory]>make_unique[PeakLiftFunctionFactoryImpl](
            num_labels, peak_label, max_lift, curvature)

