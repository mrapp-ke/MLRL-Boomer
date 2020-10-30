from boomer.common.input_data cimport LabelMatrix, IRandomAccessLabelMatrix
from boomer.common.statistics cimport StatisticsProvider, StatisticsProviderFactory, AbstractStatistics
from boomer.boosting._lapack cimport Lapack
from boomer.boosting.statistics cimport AbstractGradientStatistics
from boomer.boosting.losses_example_wise cimport ExampleWiseLoss, IExampleWiseLoss
from boomer.boosting.rule_evaluation_example_wise cimport ExampleWiseRuleEvaluationFactory, \
    IExampleWiseRuleEvaluationFactory

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/statistics_example_wise.h" namespace "boosting" nogil:

    cdef cppclass AbstractExampleWiseStatistics(AbstractGradientStatistics):

        # Functions:

        void setRuleEvaluationFactory(shared_ptr[IExampleWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr)


    cdef cppclass IExampleWiseStatisticsFactory:

        # Functions:

        unique_ptr[AbstractExampleWiseStatistics] create()


    cdef cppclass DenseExampleWiseStatisticsFactoryImpl(IExampleWiseStatisticsFactory):

        # Constructors:

        DenseExampleWiseStatisticsFactoryImpl(shared_ptr[IExampleWiseLoss] lossFunctionPtr,
                                              shared_ptr[IExampleWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr,
                                              unique_ptr[Lapack] lapackPtr,
                                              shared_ptr[IRandomAccessLabelMatrix] labelMatrixPtr) except +


cdef class ExampleWiseStatisticsFactory:

    # Attributes:

    cdef shared_ptr[IExampleWiseStatisticsFactory] statistics_factory_ptr

    # Functions:

    cdef unique_ptr[AbstractExampleWiseStatistics] create(self)


cdef class DenseExampleWiseStatisticsFactory(ExampleWiseStatisticsFactory):

    # Functions:

    cdef unique_ptr[AbstractExampleWiseStatistics] create(self)


cdef class ExampleWiseStatisticsProvider(StatisticsProvider):

    # Attributes:

    cdef ExampleWiseRuleEvaluationFactory rule_evaluation_factory

    # Functions:

    cdef AbstractStatistics* get(self)


cdef class ExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):

    # Attributes:

    cdef ExampleWiseLoss loss_function

    cdef ExampleWiseRuleEvaluationFactory default_rule_evaluation_factory

    cdef ExampleWiseRuleEvaluationFactory rule_evaluation_factory

    # Functions:

    cdef StatisticsProvider create(self, LabelMatrix label_matrix)
