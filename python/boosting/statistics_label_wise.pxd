from common.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from boosting.losses_label_wise cimport ILabelWiseLoss
from boosting.rule_evaluation_label_wise cimport ILabelWiseRuleEvaluationFactory

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/statistics/statistics_label_wise_provider.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseStatisticsProviderFactoryImpl"boosting::LabelWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        LabelWiseStatisticsProviderFactoryImpl(
            shared_ptr[ILabelWiseLoss] lossFunctionPtr,
            shared_ptr[ILabelWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            shared_ptr[ILabelWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr) except +


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
