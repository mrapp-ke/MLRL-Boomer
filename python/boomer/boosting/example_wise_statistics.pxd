from boomer.common.input_data cimport LabelMatrix, IRandomAccessLabelMatrix
from boomer.common.statistics cimport StatisticsProvider, StatisticsProviderFactory, AbstractStatistics
from boomer.boosting._lapack cimport Lapack
from boomer.boosting.statistics cimport AbstractGradientStatistics
from boomer.boosting.example_wise_losses cimport ExampleWiseLoss, IExampleWiseLoss
from boomer.boosting.example_wise_rule_evaluation cimport ExampleWiseRuleEvaluation, IExampleWiseRuleEvaluation

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_statistics.h" namespace "boosting" nogil:

    cdef cppclass AbstractExampleWiseStatistics(AbstractGradientStatistics):

        # Functions:

        void setRuleEvaluation(shared_ptr[IExampleWiseRuleEvaluation] ruleEvaluationPtr)


    cdef cppclass DenseExampleWiseStatisticsImpl(AbstractExampleWiseStatistics):

        # Constructors:

        DenseExampleWiseStatisticsImpl(shared_ptr[IExampleWiseLoss] lossFunctionPtr,
                                       shared_ptr[IExampleWiseRuleEvaluation] ruleEvaluationPtr,
                                       shared_ptr[Lapack] lapackPtr) except +


    cdef cppclass IExampleWiseStatisticsFactory:

        # Functions:

        AbstractExampleWiseStatistics* create()


    cdef cppclass DenseExampleWiseStatisticsFactoryImpl(IExampleWiseStatisticsFactory):

        # Constructors:

        DenseExampleWiseStatisticsFactoryImpl(shared_ptr[IExampleWiseLoss] lossFunctionPtr,
                                              shared_ptr[IExampleWiseRuleEvaluation] ruleEvaluationPtr,
                                              shared_ptr[Lapack] lapackPtr,
                                              shared_ptr[IRandomAccessLabelMatrix] labelMatrixPtr) except +


cdef class ExampleWiseStatisticsFactory:

    # Attributes:

    cdef shared_ptr[IExampleWiseStatisticsFactory] statistics_factory_ptr

    # Functions:

    cdef AbstractExampleWiseStatistics* create(self)


cdef class DenseExampleWiseStatisticsFactory(ExampleWiseStatisticsFactory):

    # Functions:

    cdef AbstractExampleWiseStatistics* create(self)


cdef class ExampleWiseStatisticsProvider(StatisticsProvider):

    # Attributes:

    cdef shared_ptr[AbstractExampleWiseStatistics] statistics_ptr

    cdef ExampleWiseRuleEvaluation rule_evaluation

    # Functions:

    cdef AbstractStatistics* get(self)


cdef class ExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):

    # Attributes:

    cdef ExampleWiseLoss loss_function

    cdef ExampleWiseRuleEvaluation default_rule_evaluation

    cdef ExampleWiseRuleEvaluation rule_evaluation

    # Functions:

    cdef StatisticsProvider create(self, LabelMatrix label_matrix)
