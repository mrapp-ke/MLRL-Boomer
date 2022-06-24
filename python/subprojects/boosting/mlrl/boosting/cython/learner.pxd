from mlrl.common.cython.learner cimport  IRuleLearnerConfig, RuleLearnerConfig
from mlrl.boosting.cython.head_type cimport IFixedPartialHeadConfig, IDynamicPartialHeadConfig
from mlrl.boosting.cython.post_processor cimport IConstantShrinkageConfig
from mlrl.boosting.cython.regularization cimport IManualRegularizationConfig


ctypedef double (*DdotFunction)(int* n, double* dx, int* incx, double* dy, int* incy)

ctypedef void (*DspmvFunction)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta, double* y, int* incy)

ctypedef void (*DsysvFunction)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, double* work, int* lwork, int* info)


cdef extern from "boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass IBoostingRuleLearnerConfig"boosting::IBoostingRuleLearner::IConfig"(IRuleLearnerConfig):

        # Functions:

        void useCompleteHeads()

        void useDenseStatistics()

        void useNoL1Regularization()

        void useNoL2Regularization()

        void useLabelWiseLogisticLoss()

        void useNoLabelBinning()

        void useLabelWiseClassificationPredictor()

        void useLabelWiseRegressionPredictor()

        void useLabelWiseProbabilityPredictor()


    cdef cppclass IShrinkageMixin"boosting::IBoostingRuleLearner::IShrinkageMixin":

        # Functions:

        IConstantShrinkageConfig& useConstantShrinkagePostProcessor()


    cdef cppclass IRegularizationMixin"boosting::IBoostingRuleLearner::IRegularizationMixin":

        # Functions:

        IManualRegularizationConfig& useL1Regularization()

        IManualRegularizationConfig& useL2Regularization()


    cdef cppclass INoDefaultRuleMixin"boosting::IBoostingRuleLearner::INoDefaultRuleMixin":

        # Functions:

        void useNoDefaultRule()


    cdef cppclass IPartialHeadMixin"boosting::IBoostingRuleLearner::IPartialHeadMixin":

        # Functions:

        IFixedPartialHeadConfig& useFixedPartialHeads()

        IDynamicPartialHeadConfig& useDynamicPartialHeads()

        void useSingleLabelHeads()


cdef class BoostingRuleLearnerConfig(RuleLearnerConfig):

    # Functions:

    cdef IBoostingRuleLearnerConfig* get_boosting_rule_learner_config_ptr(self)
