from boomer.common.input_data cimport LabelMatrix, IRandomAccessLabelMatrix
from boomer.common.statistics cimport StatisticsProvider, StatisticsProviderFactory, AbstractStatistics
from boomer.boosting.statistics cimport AbstractGradientStatistics
from boomer.boosting.label_wise_losses cimport LabelWiseLoss, AbstractLabelWiseLoss
from boomer.boosting.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation, AbstractLabelWiseRuleEvaluation

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_statistics.h" namespace "boosting" nogil:

    cdef cppclass AbstractLabelWiseStatistics(AbstractGradientStatistics):

        # Functions:

        void setRuleEvaluation(shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr)


    cdef cppclass DenseLabelWiseStatisticsImpl(AbstractLabelWiseStatistics):

        # Constructors:

        DenseLabelWiseStatisticsImpl(shared_ptr[AbstractLabelWiseLoss] lossFunctionPtr,
                                     shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr) except +


    cdef cppclass AbstractLabelWiseStatisticsFactory:

        # Functions:

        AbstractLabelWiseStatistics* create()


    cdef cppclass DenseLabelWiseStatisticsFactoryImpl(AbstractLabelWiseStatisticsFactory):

        # Constructors:

        DenseLabelWiseStatisticsFactoryImpl(shared_ptr[AbstractLabelWiseLoss] lossFunctionPtr,
                                            shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr,
                                            shared_ptr[IRandomAccessLabelMatrix] labelMatrixPtr) except +


cdef class LabelWiseStatisticsFactory:

    # Attributes:

    cdef shared_ptr[AbstractLabelWiseStatisticsFactory] statistics_factory_ptr

    # Functions:

    cdef AbstractLabelWiseStatistics* create(self)


cdef class DenseLabelWiseStatisticsFactory(LabelWiseStatisticsFactory):

    # Functions:

    cdef AbstractLabelWiseStatistics* create(self)


cdef class LabelWiseStatisticsProvider(StatisticsProvider):

    # Attributes:

    cdef shared_ptr[AbstractLabelWiseStatistics] statistics_ptr

    cdef LabelWiseRuleEvaluation rule_evaluation

    # Functions:

    cdef AbstractStatistics* get(self)


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):

    # Attributes:

    cdef LabelWiseLoss loss_function

    cdef LabelWiseRuleEvaluation default_rule_evaluation

    cdef LabelWiseRuleEvaluation rule_evaluation

    # Functions:

    cdef StatisticsProvider create(self, LabelMatrix label_matrix)
