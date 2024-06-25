from mlrl.boosting.cython.head_type cimport IDynamicPartialHeadConfig, IFixedPartialHeadConfig
from mlrl.boosting.cython.post_processor cimport IConstantShrinkageConfig
from mlrl.boosting.cython.regularization cimport IManualRegularizationConfig

ctypedef double (*DdotFunction)(int* n, double* dx, int* incx, double* dy, int* incy)

ctypedef void (*DspmvFunction)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta,
                               double* y, int* incy)

ctypedef void (*DsysvFunction)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb,
                               double* work, int* lwork, int* info)


cdef extern from "mlrl/boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass IAutomaticFeatureBinningMixin"boosting::IBoostedRuleLearner::IAutomaticFeatureBinningMixin":

        # Functions

        void useAutomaticFeatureBinning()


    cdef cppclass IAutomaticParallelRuleRefinementMixin\
        "boosting::IBoostedRuleLearner::IAutomaticParallelRuleRefinementMixin":

        # Functions:

        void useAutomaticParallelRuleRefinement()


    cdef cppclass IAutomaticParallelStatisticUpdateMixin\
        "boosting::IBoostedRuleLearner::IAutomaticParallelStatisticUpdateMixin":

        # Functions:

        void useAutomaticParallelStatisticUpdate()


    cdef cppclass IConstantShrinkageMixin"boosting::IBoostedRuleLearner::IConstantShrinkageMixin":

        # Functions:

        IConstantShrinkageConfig& useConstantShrinkagePostProcessor()


    cdef cppclass INoL1RegularizationMixin"boosting::IBoostedRuleLearner::INoL1RegularizationMixin":

        # Functions:

        void useNoL1Regularization()


    cdef cppclass IL1RegularizationMixin"boosting::IBoostedRuleLearner::IL1RegularizationMixin":

        # Functions:

        IManualRegularizationConfig& useL1Regularization()


    cdef cppclass INoL2RegularizationMixin"boosting::IBoostedRuleLearner::INoL2RegularizationMixin":

        # Functions:

        void useNoL2Regularization()


    cdef cppclass IL2RegularizationMixin"boosting::IBoostedRuleLearner::IL2RegularizationMixin":

        # Functions:

        IManualRegularizationConfig& useL2Regularization()


    cdef cppclass ICompleteHeadMixin"boosting::IBoostedRuleLearner::ICompleteHeadMixin":

        # Functions:

        void useCompleteHeads()


    cdef cppclass IFixedPartialHeadMixin"boosting::IBoostedRuleLearner::IFixedPartialHeadMixin":

        # Functions:

        IFixedPartialHeadConfig& useFixedPartialHeads()


    cdef cppclass IDynamicPartialHeadMixin"boosting::IBoostedRuleLearner::IDynamicPartialHeadMixin":

        # Functions:

        IDynamicPartialHeadConfig& useDynamicPartialHeads()


    cdef cppclass ISingleOutputHeadMixin"boosting::IBoostedRuleLearner::ISingleOutputHeadMixin":

        # Functions:
        
        void useSingleOutputHeads()


    cdef cppclass IAutomaticHeadMixin"boosting::IBoostedRuleLearner::IAutomaticHeadMixin":

        # Functions:

        void useAutomaticHeads()


    cdef cppclass INonDecomposableSquaredErrorLossMixin\
        "boosting::IBoostedRuleLearner::INonDecomposableSquaredErrorLossMixin":

        # Functions:

        void useNonDecomposableSquaredErrorLoss()


    cdef cppclass IDecomposableSquaredErrorLossMixin"boosting::IBoostedRuleLearner::IDecomposableSquaredErrorLossMixin":

        # Functions:

        void useDecomposableSquaredErrorLoss()


    cdef cppclass IOutputWiseScorePredictorMixin"boosting::IBoostedRuleLearner::IOutputWiseScorePredictorMixin":

        # Functions:

        void useOutputWiseScorePredictor()
