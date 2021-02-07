from boomer.common.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from boomer.seco.rule_evaluation_label_wise cimport ILabelWiseRuleEvaluationFactory

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/statistics/statistics_label_wise_provider.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseStatisticsProviderFactoryImpl"seco::LabelWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        LabelWiseStatisticsProviderFactoryImpl(
            shared_ptr[ILabelWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            shared_ptr[ILabelWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr) except +


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
