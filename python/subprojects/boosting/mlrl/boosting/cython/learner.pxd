from mlrl.boosting.cython.head_type cimport IDynamicPartialHeadConfig, IFixedPartialHeadConfig
from mlrl.boosting.cython.label_binning cimport IEqualWidthLabelBinningConfig
from mlrl.boosting.cython.post_processor cimport IConstantShrinkageConfig
from mlrl.boosting.cython.prediction cimport IExampleWiseBinaryPredictorConfig, IGfmBinaryPredictorConfig, \
    IMarginalizedProbabilityPredictorConfig, IOutputWiseBinaryPredictorConfig, IOutputWiseProbabilityPredictorConfig
from mlrl.boosting.cython.probability_calibration cimport IIsotonicJointProbabilityCalibratorConfig, \
    IIsotonicMarginalProbabilityCalibratorConfig
from mlrl.boosting.cython.regularization cimport IManualRegularizationConfig

ctypedef double (*DdotFunction)(int* n, double* dx, int* incx, double* dy, int* incy)

ctypedef void (*DspmvFunction)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta,
                               double* y, int* incy)

ctypedef void (*DsysvFunction)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb,
                               double* work, int* lwork, int* info)


cdef extern from "mlrl/boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass IAutomaticPartitionSamplingMixin"boosting::IBoostedRuleLearner::IAutomaticPartitionSamplingMixin":

        # Functions:

        void useAutomaticPartitionSampling()


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


    cdef cppclass INoDefaultRuleMixin"boosting::IBoostedRuleLearner::INoDefaultRuleMixin":

        # Functions:

        void useNoDefaultRule()


    cdef cppclass IAutomaticDefaultRuleMixin"boosting::IBoostedRuleLearner::IAutomaticDefaultRuleMixin":

        # Functions:

        void useAutomaticDefaultRule()


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


    cdef cppclass IDenseStatisticsMixin"boosting::IBoostedRuleLearner::IDenseStatisticsMixin":

        # Functions:

        void useDenseStatistics()


    cdef cppclass ISparseStatisticsMixin"boosting::IBoostedRuleLearner::ISparseStatisticsMixin":

        # Functions:

        void useSparseStatistics()


    cdef cppclass IAutomaticStatisticsMixin"boosting::IBoostedRuleLearner::IAutomaticStatisticsMixin":

        # Functions:

        void useAutomaticStatistics()


    cdef cppclass INonDecomposableLogisticLossMixin"boosting::IBoostedRuleLearner::INonDecomposableLogisticLossMixin":

        # Functions:

        void useNonDecomposableLogisticLoss()


    cdef cppclass INonDecomposableSquaredErrorLossMixin"boosting::IBoostedRuleLearner::INonDecomposableSquaredErrorLossMixin":

        # Functions:

        void useNonDecomposableSquaredErrorLoss()


    cdef cppclass INonDecomposableSquaredHingeLossMixin"boosting::IBoostedRuleLearner::INonDecomposableSquaredHingeLossMixin":

        # Functions:

        void useNonDecomposableSquaredHingeLoss()


    cdef cppclass IDecomposableLogisticLossMixin"boosting::IBoostedRuleLearner::IDecomposableLogisticLossMixin":

        # Functions:

        void useDecomposableLogisticLoss()

        
    cdef cppclass IDecomposableSquaredErrorLossMixin"boosting::IBoostedRuleLearner::IDecomposableSquaredErrorLossMixin":

        # Functions:

        void useDecomposableSquaredErrorLoss()


    cdef cppclass IDecomposableSquaredHingeLossMixin"boosting::IBoostedRuleLearner::IDecomposableSquaredHingeLossMixin":

        # Functions:

        void useDecomposableSquaredHingeLoss()


    cdef cppclass INoLabelBinningMixin"boosting::IBoostedRuleLearner::INoLabelBinningMixin":

        # Functions:

        void useNoLabelBinning()


    cdef cppclass IEqualWidthLabelBinningMixin"boosting::IBoostedRuleLearner::IEqualWidthLabelBinningMixin":

        # Functions:

        IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning()


    cdef cppclass IIsotonicMarginalProbabilityCalibrationMixin"boosting::IBoostedRuleLearner::IIsotonicMarginalProbabilityCalibrationMixin":

        # Functions:

        IIsotonicMarginalProbabilityCalibratorConfig& useIsotonicMarginalProbabilityCalibration()


    cdef cppclass IIsotonicJointProbabilityCalibrationMixin"boosting::IBoostedRuleLearner::IIsotonicJointProbabilityCalibrationMixin":

        # Functions:

        IIsotonicJointProbabilityCalibratorConfig& useIsotonicJointProbabilityCalibration()
        

    cdef cppclass IAutomaticLabelBinningMixin"boosting::IBoostedRuleLearner::IAutomaticLabelBinningMixin":

        # Functions:

        void useAutomaticLabelBinning()


    cdef cppclass IOutputWiseBinaryPredictorMixin"boosting::IBoostedRuleLearner::IOutputWiseBinaryPredictorMixin":

        # Functions:

        IOutputWiseBinaryPredictorConfig& useOutputWiseBinaryPredictor()


    cdef cppclass IExampleWiseBinaryPredictorMixin"boosting::IBoostedRuleLearner::IExampleWiseBinaryPredictorMixin":

        # Functions:

        IExampleWiseBinaryPredictorConfig& useExampleWiseBinaryPredictor()


    cdef cppclass IGfmBinaryPredictorMixin"boosting::IBoostedRuleLearner::IGfmBinaryPredictorMixin":

        # Functions:

        IGfmBinaryPredictorConfig& useGfmBinaryPredictor()


    cdef cppclass IAutomaticBinaryPredictorMixin"boosting::IBoostedRuleLearner::IAutomaticBinaryPredictorMixin":

        # Functions:

        void useAutomaticBinaryPredictor()


    cdef cppclass IOutputWiseScorePredictorMixin"boosting::IBoostedRuleLearner::IOutputWiseScorePredictorMixin":

        # Functions:

        void useOutputWiseScorePredictor()


    cdef cppclass IOutputWiseProbabilityPredictorMixin \
        "boosting::IBoostedRuleLearner::IOutputWiseProbabilityPredictorMixin":

        # Functions:

        IOutputWiseProbabilityPredictorConfig& useOutputWiseProbabilityPredictor()


    cdef cppclass IMarginalizedProbabilityPredictorMixin\
        "boosting::IBoostedRuleLearner::IMarginalizedProbabilityPredictorMixin":

        # Functions:

        IMarginalizedProbabilityPredictorConfig& useMarginalizedProbabilityPredictor()


    cdef cppclass IAutomaticProbabilityPredictorMixin\
        "boosting::IBoostedRuleLearner::IAutomaticProbabilityPredictorMixin":
        
        # Functions:

        void useAutomaticProbabilityPredictor()
