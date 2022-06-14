from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, IRuleLearnerConfig, RuleLearnerConfig, \
    IBeamSearchTopDownMixin, IFeatureBinningMixin, ILabelSamplingMixin, IInstanceSamplingMixin, IFeatureSamplingMixin, \
    IPartitionSamplingMixin, IPruningMixin, IMultiThreadingMixin, ISizeStoppingCriterionMixin, \
    ITimeStoppingCriterionMixin, IMeasureStoppingCriterionMixin
from mlrl.boosting.cython.head_type cimport IFixedPartialHeadConfig, IDynamicPartialHeadConfig
from mlrl.boosting.cython.label_binning cimport IEqualWidthLabelBinningConfig
from mlrl.boosting.cython.post_processor cimport IConstantShrinkageConfig
from mlrl.boosting.cython.regularization cimport IManualRegularizationConfig

from libcpp.memory cimport unique_ptr


ctypedef double (*DdotFunction)(int* n, double* dx, int* incx, double* dy, int* incy)

ctypedef void (*DspmvFunction)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta, double* y, int* incy)

ctypedef void (*DsysvFunction)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, double* work, int* lwork, int* info)


cdef extern from "boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass IBoomerConfig"boosting::IBoomer::IConfig"(IRuleLearnerConfig,
                                                            IBeamSearchTopDownMixin,
                                                            IFeatureBinningMixin,
                                                            ILabelSamplingMixin,
                                                            IInstanceSamplingMixin,
                                                            IFeatureSamplingMixin,
                                                            IPartitionSamplingMixin,
                                                            IPruningMixin,
                                                            IMultiThreadingMixin,
                                                            ISizeStoppingCriterionMixin,
                                                            ITimeStoppingCriterionMixin,
                                                            IMeasureStoppingCriterionMixin):

        # Functions:

        void useNoDefaultRule()

        void useAutomaticDefaultRule()

        void useAutomaticFeatureBinning()

        IConstantShrinkageConfig& useConstantShrinkagePostProcessor()

        void useAutomaticParallelRuleRefinement()

        void useAutomaticParallelStatisticUpdate()

        void useAutomaticHeads()

        void useSingleLabelHeads()

        IFixedPartialHeadConfig& useFixedPartialHeads()

        IDynamicPartialHeadConfig& useDynamicPartialHeads()

        void useCompleteHeads()

        void useAutomaticStatistics()

        void useDenseStatistics()

        void useSparseStatistics()

        void useNoL1Regularization()

        IManualRegularizationConfig& useL1Regularization()

        void useNoL2Regularization()

        IManualRegularizationConfig& useL2Regularization()

        void useExampleWiseLogisticLoss()

        void useLabelWiseLogisticLoss()

        void useLabelWiseSquaredErrorLoss()

        void useLabelWiseSquaredHingeLoss()

        void useNoLabelBinning()

        void useAutomaticLabelBinning()

        IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning()

        void useExampleWiseClassificationPredictor()

        void useLabelWiseClassificationPredictor()

        void useLabelWiseRegressionPredictor()

        void useLabelWiseProbabilityPredictor()

        void useMarginalizedProbabilityPredictor()

        void useAutomaticProbabilityPredictor()


    cdef cppclass IBoomer(IRuleLearner):
        pass


    unique_ptr[IBoomerConfig] createBoomerConfig()


    unique_ptr[IBoomer] createBoomer(unique_ptr[IBoomerConfig] configPtr, DdotFunction ddotFunction,
                                     DspmvFunction dspmvFunction, DsysvFunction dsysvFunction)


cdef class BoomerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoomerConfig] rule_learner_config_ptr


cdef class Boomer(RuleLearner):

    # Attributes:

    cdef unique_ptr[IBoomer] rule_learner_ptr
