from mlrl.common.cython.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from mlrl.boosting.cython.losses_example_wise cimport IExampleWiseLoss
from mlrl.boosting.cython.rule_evaluation_example_wise cimport IExampleWiseRuleEvaluationFactory

from libcpp.memory cimport shared_ptr


cdef extern from "boosting/statistics/statistics_example_wise_dense.hpp" namespace "boosting" nogil:

    cdef cppclass DenseExampleWiseStatisticsProviderFactoryImpl"boosting::DenseExampleWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        DenseExampleWiseStatisticsProviderFactoryImpl(
            shared_ptr[IExampleWiseLoss] lossFunctionPtr,
            shared_ptr[IExampleWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            shared_ptr[IExampleWiseRuleEvaluationFactory] regularRuleEvaluationFactoryPtr) except +


cdef class DenseExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
