from common.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from boosting.losses_example_wise cimport IExampleWiseLoss
from boosting.rule_evaluation_example_wise cimport IExampleWiseRuleEvaluationFactory

from libcpp.memory cimport shared_ptr


cdef extern from "boosting/statistics/statistics_example_wise_provider.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseStatisticsProviderFactoryImpl"boosting::ExampleWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        ExampleWiseStatisticsProviderFactoryImpl(
            shared_ptr[IExampleWiseLoss] lossFunctionPtr,
            shared_ptr[IExampleWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            shared_ptr[IExampleWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr) except +


cdef class ExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
