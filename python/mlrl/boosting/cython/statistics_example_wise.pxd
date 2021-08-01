from mlrl.common.cython.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from mlrl.boosting.cython.losses_example_wise cimport IExampleWiseLoss
from mlrl.boosting.cython.rule_evaluation_example_wise cimport IExampleWiseRuleEvaluationFactory

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/statistics/statistics_example_wise_dense.hpp" namespace "boosting" nogil:

    cdef cppclass DenseExampleWiseStatisticsProviderFactoryImpl"boosting::DenseExampleWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        DenseExampleWiseStatisticsProviderFactoryImpl(
            unique_ptr[IExampleWiseLoss] lossFunctionPtr,
            unique_ptr[IExampleWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            unique_ptr[IExampleWiseRuleEvaluationFactory] regularRuleEvaluationFactoryPtr,
            unique_ptr[IExampleWiseRuleEvaluationFactory] pruningRuleEvaluationFactoryPtr) except +


cdef class DenseExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
