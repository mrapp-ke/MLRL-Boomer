from boomer.common.input_data cimport LabelMatrix, IRandomAccessLabelMatrix
from boomer.common.statistics cimport StatisticsProvider, StatisticsProviderFactory, AbstractStatistics
from boomer.seco.statistics cimport AbstractCoverageStatistics
from boomer.seco.label_wise_rule_evaluation cimport LabelWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/label_wise_statistics.h" namespace "seco" nogil:

    cdef cppclass AbstractLabelWiseStatistics(AbstractCoverageStatistics):

        # Functions:

        void setRuleEvaluationFactory(shared_ptr[ILabelWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr)


    cdef cppclass DenseLabelWiseStatisticsImpl(AbstractLabelWiseStatistics):

        # Constructors:

        DenseLabelWiseStatisticsImpl(shared_ptr[ILabelWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr) except +


    cdef cppclass ILabelWiseStatisticsFactory:

        # Functions:

        unique_ptr[AbstractLabelWiseStatistics] create()


    cdef cppclass DenseLabelWiseStatisticsFactoryImpl(ILabelWiseStatisticsFactory):

        # Constructors:

        DenseLabelWiseStatisticsFactoryImpl(shared_ptr[ILabelWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr,
                                            shared_ptr[IRandomAccessLabelMatrix] labelMatrixPtr) except +


cdef class LabelWiseStatisticsFactory:

    # Attributes:

    cdef shared_ptr[ILabelWiseStatisticsFactory] statistics_factory_ptr

    # Functions:

    cdef unique_ptr[AbstractLabelWiseStatistics] create(self)


cdef class DenseLabelWiseStatisticsFactory(LabelWiseStatisticsFactory):

    # Functions:

    cdef unique_ptr[AbstractLabelWiseStatistics] create(self)


cdef class LabelWiseStatisticsProvider(StatisticsProvider):

    # Attributes:

    cdef LabelWiseRuleEvaluationFactory rule_evaluation_factory

    # Functions:

    cdef AbstractStatistics* get(self)


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):

    # Attributes:

    cdef LabelWiseRuleEvaluationFactory default_rule_evaluation_factory

    cdef LabelWiseRuleEvaluationFactory rule_evaluation_factory

    # Functions:

    cdef StatisticsProvider create(self, LabelMatrix label_matrix)
