from common.cython._measures cimport IMeasure
from common.cython.measures cimport Measure

from libcpp.memory cimport shared_ptr


cdef extern from "boosting/losses/loss_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseLoss(IMeasure):
        pass


cdef extern from "boosting/losses/loss_example_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseLogisticLossImpl"boosting::ExampleWiseLogisticLoss"(IExampleWiseLoss):
        pass


cdef class ExampleWiseLoss(Measure):

    # Attributes:

    cdef shared_ptr[IExampleWiseLoss] loss_function_ptr


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    pass
