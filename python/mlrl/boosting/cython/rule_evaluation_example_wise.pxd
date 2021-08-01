from mlrl.common.cython._types cimport uint32, float32, float64
from mlrl.boosting.cython._blas cimport Blas
from mlrl.boosting.cython._lapack cimport Lapack
from mlrl.boosting.cython.binning cimport ILabelBinningFactory

from libcpp.memory cimport shared_ptr, unique_ptr


cdef extern from "boosting/rule_evaluation/rule_evaluation_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseRuleEvaluationFactory:
        pass


cdef extern from "boosting/rule_evaluation/rule_evaluation_example_wise_regularized.hpp" namespace "boosting" nogil:

    cdef cppclass RegularizedExampleWiseRuleEvaluationFactoryImpl"boosting::RegularizedExampleWiseRuleEvaluationFactory"(
            IExampleWiseRuleEvaluationFactory):

        # Constructors:

        RegularizedExampleWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight, unique_ptr[Blas] blasPtr,
                                                        unique_ptr[Lapack] lapackPtr) except +


cdef extern from "boosting/rule_evaluation/rule_evaluation_example_wise_binning.hpp" namespace "boosting" nogil:

    cdef cppclass BinnedExampleWiseRuleEvaluationFactoryImpl"boosting::BinnedExampleWiseRuleEvaluationFactory"(
            IExampleWiseRuleEvaluationFactory):

        # Constructors:

        BinnedExampleWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight,
                                                   unique_ptr[ILabelBinningFactory] labelBinningFactoryPtr,
                                                   unique_ptr[Blas] blasPtr, unique_ptr[Lapack] lapackPtr) except +


cdef class ExampleWiseRuleEvaluationFactory:

    # Attributes:

    cdef shared_ptr[IExampleWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class RegularizedExampleWiseRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    pass


cdef class BinnedExampleWiseRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    pass
