from mlrl.common.cython.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from mlrl.boosting.cython.losses_label_wise cimport ILabelWiseLoss
from mlrl.boosting.cython.rule_evaluation_label_wise cimport ILabelWiseRuleEvaluationFactory

from libcpp.memory cimport shared_ptr


cdef extern from "boosting/statistics/statistics_label_wise_dense.hpp" namespace "boosting" nogil:

    cdef cppclass DenseLabelWiseStatisticsProviderFactoryImpl"boosting::DenseLabelWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        DenseLabelWiseStatisticsProviderFactoryImpl(
            shared_ptr[ILabelWiseLoss] lossFunctionPtr,
            shared_ptr[ILabelWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            shared_ptr[ILabelWiseRuleEvaluationFactory] regularRuleEvaluationFactoryPtr,
            shared_ptr[ILabelWiseRuleEvaluationFactory] pruningRuleEvaluationFactoryPtr) except +


cdef class DenseLabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
