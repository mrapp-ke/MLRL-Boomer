from common._measures cimport IMeasure

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/losses/loss_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseLoss(IMeasure):
        pass


cdef extern from "cpp/losses/loss_example_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseLogisticLossImpl"boosting::ExampleWiseLogisticLoss"(IExampleWiseLoss):
        pass


cdef class ExampleWiseLoss:

    # Attributes:

    cdef shared_ptr[IExampleWiseLoss] loss_function_ptr


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    pass
