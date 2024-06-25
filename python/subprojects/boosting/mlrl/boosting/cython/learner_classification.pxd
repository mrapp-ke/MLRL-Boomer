from mlrl.boosting.cython.label_binning cimport IEqualWidthLabelBinningConfig
from mlrl.boosting.cython.prediction cimport IExampleWiseBinaryPredictorConfig, IGfmBinaryPredictorConfig, \
    IMarginalizedProbabilityPredictorConfig, IOutputWiseBinaryPredictorConfig, IOutputWiseProbabilityPredictorConfig
from mlrl.boosting.cython.probability_calibration cimport IIsotonicJointProbabilityCalibratorConfig, \
    IIsotonicMarginalProbabilityCalibratorConfig


cdef extern from "mlrl/boosting/learner_classification.hpp" namespace "boosting" nogil:

    cdef cppclass IAutomaticPartitionSamplingMixin\
        "boosting::IBoostedClassificationRuleLearner::IAutomaticPartitionSamplingMixin":

        # Functions:

        void useAutomaticPartitionSampling()


    cdef cppclass INoDefaultRuleMixin"boosting::IBoostedClassificationRuleLearner::INoDefaultRuleMixin":

        # Functions:

        void useNoDefaultRule()


    cdef cppclass IAutomaticDefaultRuleMixin"boosting::IBoostedClassificationRuleLearner::IAutomaticDefaultRuleMixin":

        # Functions:

        void useAutomaticDefaultRule()


    cdef cppclass IDenseStatisticsMixin"boosting::IBoostedClassificationRuleLearner::IDenseStatisticsMixin":

        # Functions:

        void useDenseStatistics()


    cdef cppclass ISparseStatisticsMixin"boosting::IBoostedClassificationRuleLearner::ISparseStatisticsMixin":

        # Functions:

        void useSparseStatistics()


    cdef cppclass IAutomaticStatisticsMixin"boosting::IBoostedClassificationRuleLearner::IAutomaticStatisticsMixin":

        # Functions:

        void useAutomaticStatistics()


    cdef cppclass INonDecomposableLogisticLossMixin\
        "boosting::IBoostedClassificationRuleLearner::INonDecomposableLogisticLossMixin":

        # Functions:

        void useNonDecomposableLogisticLoss()


    cdef cppclass INonDecomposableSquaredHingeLossMixin\
        "boosting::IBoostedClassificationRuleLearner::INonDecomposableSquaredHingeLossMixin":

        # Functions:

        void useNonDecomposableSquaredHingeLoss()


    cdef cppclass IDecomposableLogisticLossMixin\
        "boosting::IBoostedClassificationRuleLearner::IDecomposableLogisticLossMixin":

        # Functions:

        void useDecomposableLogisticLoss()

        
    cdef cppclass IDecomposableSquaredHingeLossMixin\
        "boosting::IBoostedClassificationRuleLearner::IDecomposableSquaredHingeLossMixin":

        # Functions:

        void useDecomposableSquaredHingeLoss()


    cdef cppclass INoLabelBinningMixin"boosting::IBoostedClassificationRuleLearner::INoLabelBinningMixin":

        # Functions:

        void useNoLabelBinning()


    cdef cppclass IEqualWidthLabelBinningMixin\
        "boosting::IBoostedClassificationRuleLearner::IEqualWidthLabelBinningMixin":

        # Functions:

        IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning()


    cdef cppclass IAutomaticLabelBinningMixin"boosting::IBoostedClassificationRuleLearner::IAutomaticLabelBinningMixin":

        # Functions:

        void useAutomaticLabelBinning()


    cdef cppclass IIsotonicMarginalProbabilityCalibrationMixin\
        "boosting::IBoostedClassificationRuleLearner::IIsotonicMarginalProbabilityCalibrationMixin":

        # Functions:

        IIsotonicMarginalProbabilityCalibratorConfig& useIsotonicMarginalProbabilityCalibration()


    cdef cppclass IIsotonicJointProbabilityCalibrationMixin\
        "boosting::IBoostedClassificationRuleLearner::IIsotonicJointProbabilityCalibrationMixin":

        # Functions:

        IIsotonicJointProbabilityCalibratorConfig& useIsotonicJointProbabilityCalibration()
        

    cdef cppclass IOutputWiseProbabilityPredictorMixin \
        "boosting::IBoostedClassificationRuleLearner::IOutputWiseProbabilityPredictorMixin":

        # Functions:

        IOutputWiseProbabilityPredictorConfig& useOutputWiseProbabilityPredictor()


    cdef cppclass IMarginalizedProbabilityPredictorMixin\
        "boosting::IBoostedClassificationRuleLearner::IMarginalizedProbabilityPredictorMixin":

        # Functions:

        IMarginalizedProbabilityPredictorConfig& useMarginalizedProbabilityPredictor()


    cdef cppclass IAutomaticProbabilityPredictorMixin\
        "boosting::IBoostedClassificationRuleLearner::IAutomaticProbabilityPredictorMixin":
        
        # Functions:

        void useAutomaticProbabilityPredictor()


    cdef cppclass IOutputWiseBinaryPredictorMixin\
        "boosting::IBoostedClassificationRuleLearner::IOutputWiseBinaryPredictorMixin":

        # Functions:

        IOutputWiseBinaryPredictorConfig& useOutputWiseBinaryPredictor()


    cdef cppclass IExampleWiseBinaryPredictorMixin\
        "boosting::IBoostedClassificationRuleLearner::IExampleWiseBinaryPredictorMixin":

        # Functions:

        IExampleWiseBinaryPredictorConfig& useExampleWiseBinaryPredictor()


    cdef cppclass IGfmBinaryPredictorMixin"boosting::IBoostedClassificationRuleLearner::IGfmBinaryPredictorMixin":

        # Functions:

        IGfmBinaryPredictorConfig& useGfmBinaryPredictor()


    cdef cppclass IAutomaticBinaryPredictorMixin\
        "boosting::IBoostedClassificationRuleLearner::IAutomaticBinaryPredictorMixin":

        # Functions:

        void useAutomaticBinaryPredictor()
