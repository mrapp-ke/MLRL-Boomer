from mlrl.common.cython._measures cimport IEvaluationMeasure, ISparseEvaluationMeasure
from mlrl.common.cython.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from mlrl.boosting.cython.losses_label_wise cimport ILabelWiseLoss, ISparseLabelWiseLoss
from mlrl.boosting.cython.rule_evaluation_label_wise cimport ILabelWiseRuleEvaluationFactory

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/statistics/statistics_provider_factory_label_wise_dense.hpp" namespace "boosting" nogil:

    cdef cppclass DenseLabelWiseStatisticsProviderFactoryImpl"boosting::DenseLabelWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        DenseLabelWiseStatisticsProviderFactoryImpl(
            unique_ptr[ILabelWiseLoss] lossFunctionPtr, unique_ptr[IEvaluationMeasure] evaluationMeasurePtr,
            unique_ptr[ILabelWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            unique_ptr[ILabelWiseRuleEvaluationFactory] regularRuleEvaluationFactoryPtr,
            unique_ptr[ILabelWiseRuleEvaluationFactory] pruningRuleEvaluationFactoryPtr) except +


cdef extern from "boosting/statistics/statistics_provider_factory_label_wise_sparse.hpp" namespace "boosting" nogil:

    cdef cppclass SparseLabelWiseStatisticsProviderFactoryImpl"boosting::SparseLabelWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        SparseLabelWiseStatisticsProviderFactoryImpl(
            unique_ptr[ISparseLabelWiseLoss] lossFunctionPtr, unique_ptr[ISparseEvaluationMeasure] evaluationMeasurePtr,
            unique_ptr[ILabelWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            unique_ptr[ILabelWiseRuleEvaluationFactory] regularRuleEvaluationFactoryPtr,
            unique_ptr[ILabelWiseRuleEvaluationFactory] pruningRuleEvaluationFactoryPtr) except +


cdef class DenseLabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass


cdef class SparseLabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
