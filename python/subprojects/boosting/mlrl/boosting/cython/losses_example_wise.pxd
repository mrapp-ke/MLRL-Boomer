from mlrl.common.cython._measures cimport IEvaluationMeasureFactory, ISimilarityMeasureFactory
from mlrl.common.cython.measures cimport EvaluationMeasureFactory

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/losses/loss_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseLossFactory(IEvaluationMeasureFactory, ISimilarityMeasureFactory):
        pass


cdef extern from "boosting/losses/loss_example_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseLogisticLossFactoryImpl"boosting::ExampleWiseLogisticLossFactory"(IExampleWiseLossFactory):
        pass


cdef class ExampleWiseLossFactory(EvaluationMeasureFactory):

    # Attributes:

    cdef unique_ptr[IExampleWiseLossFactory] loss_factory_ptr


cdef class ExampleWiseLogisticLossFactory(ExampleWiseLossFactory):
    pass
