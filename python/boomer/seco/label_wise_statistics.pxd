from boomer.common._arrays cimport intp, uint8, float64
from boomer.common.input_data cimport LabelMatrix, AbstractRandomAccessLabelMatrix
from boomer.common.statistics cimport StatisticsProvider, StatisticsProviderFactory, AbstractStatistics, \
    AbstractDecomposableRefinementSearch
from boomer.seco.statistics cimport AbstractCoverageStatistics
from boomer.seco.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation, AbstractLabelWiseRuleEvaluation

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_statistics.h" namespace "seco" nogil:

    cdef cppclass DenseLabelWiseRefinementSearchImpl(AbstractDecomposableRefinementSearch):

        # Constructors:

        DenseLabelWiseRefinementSearchImpl(shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr, intp numLabels,
                                          const intp* labelIndices,
                                          shared_ptr[AbstractRandomAccessLabelMatrix] labelMatrixPtr,
                                          const float64* uncoveredLabels, const uint8* minorityLabels,
                                          const float64* confusionMatricesTotal,
                                          const float64* confusionMatricesSubset) except +


    cdef cppclass AbstractLabelWiseStatistics(AbstractCoverageStatistics):

        # Functions:

        void setRuleEvaluation(shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr)


    cdef cppclass DenseLabelWiseStatisticsImpl(AbstractLabelWiseStatistics):

        # Constructors:

        DenseLabelWiseStatisticsImpl(shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr) except +


    cdef cppclass AbstractLabelWiseStatisticsFactory:

        # Functions:

        AbstractLabelWiseStatistics* create()


    cdef cppclass DenseLabelWiseStatisticsFactoryImpl(AbstractLabelWiseStatisticsFactory):

        # Constructors:

        DenseLabelWiseStatisticsFactoryImpl(shared_ptr[AbstractLabelWiseRuleEvaluation] ruleEvaluationPtr,
                                            shared_ptr[AbstractRandomAccessLabelMatrix] labelMatrixPtr) except +


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

    cdef LabelWiseRuleEvaluation default_rule_evaluation

    cdef LabelWiseRuleEvaluation rule_evaluation

    # Functions:

    cdef StatisticsProvider create(self, LabelMatrix label_matrix)
