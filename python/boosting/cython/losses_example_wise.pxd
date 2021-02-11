from common.cython._measures cimport ISimilarityMeasure
from common.cython.measures cimport SimilarityMeasure

from libcpp.memory cimport shared_ptr


cdef extern from "boosting/losses/loss_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseLoss(ISimilarityMeasure):
        pass


cdef extern from "boosting/losses/loss_example_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseLogisticLossImpl"boosting::ExampleWiseLogisticLoss"(IExampleWiseLoss):
        pass


cdef class ExampleWiseLoss(SimilarityMeasure):

    # Attributes:

    cdef shared_ptr[IExampleWiseLoss] loss_function_ptr


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    pass
