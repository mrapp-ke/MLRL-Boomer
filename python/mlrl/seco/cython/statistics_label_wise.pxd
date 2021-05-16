from mlrl.common.cython.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from mlrl.seco.cython.rule_evaluation_label_wise cimport ILabelWiseRuleEvaluationFactory

from libcpp.memory cimport shared_ptr


cdef extern from "seco/statistics/statistics_label_wise_dense.hpp" namespace "seco" nogil:

    cdef cppclass DenseLabelWiseStatisticsProviderFactoryImpl"seco::DenseLabelWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        DenseLabelWiseStatisticsProviderFactoryImpl(
            shared_ptr[ILabelWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            shared_ptr[ILabelWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr) except +


cdef class DenseLabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
