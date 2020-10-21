from boomer.common._arrays cimport float64
from boomer.boosting._blas cimport Blas
from boomer.boosting._lapack cimport Lapack

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass IExampleWiseRuleEvaluationFactory:
        pass


    cdef cppclass RegularizedExampleWiseRuleEvaluationFactoryImpl(IExampleWiseRuleEvaluationFactory):

        # Constructors:

        RegularizedExampleWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight, shared_ptr[Blas] blasPtr,
                                                        shared_ptr[Lapack] lapackPtr) except +


cdef class ExampleWiseRuleEvaluationFactory:

    # Attributes:

    cdef shared_ptr[IExampleWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class RegularizedExampleWiseRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    pass
