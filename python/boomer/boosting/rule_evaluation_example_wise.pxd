from boomer.common._types cimport uint32, float64
from boomer.boosting._blas cimport Blas
from boomer.boosting._lapack cimport Lapack

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/rule_evaluation/rule_evaluation_example_wise.h" namespace "boosting" nogil:

    cdef cppclass IExampleWiseRuleEvaluationFactory:
        pass


cdef extern from "cpp/rule_evaluation/rule_evaluation_example_wise_regularized.h" namespace "boosting" nogil:

    cdef cppclass RegularizedExampleWiseRuleEvaluationFactoryImpl"boosting::RegularizedExampleWiseRuleEvaluationFactory"(
            IExampleWiseRuleEvaluationFactory):

        # Constructors:

        RegularizedExampleWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight, shared_ptr[Blas] blasPtr,
                                                        shared_ptr[Lapack] lapackPtr) except +


    cdef cppclass BinningExampleWiseRuleEvaluationFactoryImpl(IExampleWiseRuleEvaluationFactory):

        # Constructors:

        BinningExampleWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight, uint32 numBins,
                                                    shared_ptr[Blas] blasPtr, shared_ptr[Lapack] lapackPtr) except +


cdef class ExampleWiseRuleEvaluationFactory:

    # Attributes:

    cdef shared_ptr[IExampleWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class RegularizedExampleWiseRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    pass


cdef class BinningExampleWiseRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    pass
