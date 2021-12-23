"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport unique_ptr, make_unique


cdef class CoverageStoppingCriterionFactory(StoppingCriterionFactory):
    """
    A wrapper for the C++ class `CoverageStoppingCriterion`.
    """

    def __cinit__(self, float64 threshold):
        """
        :param threshold: The threshold
        """
        self.stopping_criterion_factory_ptr = <unique_ptr[IStoppingCriterionFactory]>make_unique[CoverageStoppingCriterionFactoryImpl](
            threshold)
