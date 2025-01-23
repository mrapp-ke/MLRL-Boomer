from mlrl.common.cython._types cimport float32, float64

from mlrl.boosting.cython.head_type cimport IDynamicPartialHeadConfig, IFixedPartialHeadConfig
from mlrl.boosting.cython.post_processor cimport IConstantShrinkageConfig
from mlrl.boosting.cython.regularization cimport IManualRegularizationConfig

ctypedef float32 (*SdotFunction)(int* n, float32* x, int* incx, float32* y, int* incy)

ctypedef float64 (*DdotFunction)(int* n, float64* x, int* incx, float64* y, int* incy)

ctypedef void (*SspmvFunction)(char* uplo, int* n, float32* alpha, float32* ap, float32* x, int* incx, float32* beta,
                               float32* y, int* incy)

ctypedef void (*DspmvFunction)(char* uplo, int* n, float64* alpha, float64* ap, float64* x, int* incx, float64* beta,
                               float64* y, int* incy)

ctypedef void (*SsysvFunction)(char* uplo, int* n, int* nrhs, float32* a, int* lda, int* ipiv, float32* b, int* ldb,
                               float32* work, int* lwork, int* info)

ctypedef void (*DsysvFunction)(char* uplo, int* n, int* nrhs, float64* a, int* lda, int* ipiv, float64* b, int* ldb,
                               float64* work, int* lwork, int* info)

cdef extern from "mlrl/boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass IAutomaticFeatureBinningMixin:

        # Functions

        void useAutomaticFeatureBinning()


    cdef cppclass IAutomaticParallelRuleRefinementMixin:

        # Functions:

        void useAutomaticParallelRuleRefinement()


    cdef cppclass IAutomaticParallelStatisticUpdateMixin:

        # Functions:

        void useAutomaticParallelStatisticUpdate()


    cdef cppclass IConstantShrinkageMixin:

        # Functions:

        IConstantShrinkageConfig& useConstantShrinkagePostProcessor()


    cdef cppclass IFloat32StatisticsMixin:

        # Functions:

        void use32BitStatistics()


    cdef cppclass IFloat64StatisticsMixin:

        # Functions:

        void use64BitStatistics()


    cdef cppclass INoL1RegularizationMixin:

        # Functions:

        void useNoL1Regularization()


    cdef cppclass IL1RegularizationMixin:

        # Functions:

        IManualRegularizationConfig& useL1Regularization()


    cdef cppclass INoL2RegularizationMixin:

        # Functions:

        void useNoL2Regularization()


    cdef cppclass IL2RegularizationMixin:

        # Functions:

        IManualRegularizationConfig& useL2Regularization()


    cdef cppclass ICompleteHeadMixin:

        # Functions:

        void useCompleteHeads()


    cdef cppclass IFixedPartialHeadMixin:

        # Functions:

        IFixedPartialHeadConfig& useFixedPartialHeads()


    cdef cppclass IDynamicPartialHeadMixin:

        # Functions:

        IDynamicPartialHeadConfig& useDynamicPartialHeads()


    cdef cppclass ISingleOutputHeadMixin:

        # Functions:
        
        void useSingleOutputHeads()


    cdef cppclass IAutomaticHeadMixin:

        # Functions:

        void useAutomaticHeads()


    cdef cppclass INonDecomposableSquaredErrorLossMixin:

        # Functions:

        void useNonDecomposableSquaredErrorLoss()


    cdef cppclass IDecomposableSquaredErrorLossMixin:

        # Functions:

        void useDecomposableSquaredErrorLoss()


    cdef cppclass IOutputWiseScorePredictorMixin:

        # Functions:

        void useOutputWiseScorePredictor()
