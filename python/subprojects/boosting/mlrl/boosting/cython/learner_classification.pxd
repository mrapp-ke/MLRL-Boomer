from mlrl.boosting.cython.label_binning cimport IEqualWidthLabelBinningConfig
from mlrl.boosting.cython.prediction cimport IExampleWiseBinaryPredictorConfig, IGfmBinaryPredictorConfig, \
    IMarginalizedProbabilityPredictorConfig, IOutputWiseBinaryPredictorConfig, IOutputWiseProbabilityPredictorConfig
from mlrl.boosting.cython.probability_calibration cimport IIsotonicJointProbabilityCalibratorConfig, \
    IIsotonicMarginalProbabilityCalibratorConfig


cdef extern from "mlrl/boosting/learner_classification.hpp" namespace "boosting" nogil:

    cdef cppclass IAutomaticPartitionSamplingMixin:

        # Functions:

        void useAutomaticPartitionSampling()


    cdef cppclass INoDefaultRuleMixin:

        # Functions:

        void useNoDefaultRule()


    cdef cppclass IAutomaticDefaultRuleMixin:

        # Functions:

        void useAutomaticDefaultRule()


    cdef cppclass IDenseStatisticsMixin:

        # Functions:

        void useDenseStatistics()


    cdef cppclass ISparseStatisticsMixin:

        # Functions:

        void useSparseStatistics()


    cdef cppclass IAutomaticStatisticsMixin:

        # Functions:

        void useAutomaticStatistics()


    cdef cppclass INonDecomposableLogisticLossMixin:

        # Functions:

        void useNonDecomposableLogisticLoss()


    cdef cppclass INonDecomposableSquaredHingeLossMixin:

        # Functions:

        void useNonDecomposableSquaredHingeLoss()


    cdef cppclass IDecomposableLogisticLossMixin:

        # Functions:

        void useDecomposableLogisticLoss()

        
    cdef cppclass IDecomposableSquaredHingeLossMixin:

        # Functions:

        void useDecomposableSquaredHingeLoss()


    cdef cppclass INoLabelBinningMixin:

        # Functions:

        void useNoLabelBinning()


    cdef cppclass IEqualWidthLabelBinningMixin:

        # Functions:

        IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning()


    cdef cppclass IAutomaticLabelBinningMixin:

        # Functions:

        void useAutomaticLabelBinning()


    cdef cppclass IIsotonicMarginalProbabilityCalibrationMixin:

        # Functions:

        IIsotonicMarginalProbabilityCalibratorConfig& useIsotonicMarginalProbabilityCalibration()


    cdef cppclass IIsotonicJointProbabilityCalibrationMixin:

        # Functions:

        IIsotonicJointProbabilityCalibratorConfig& useIsotonicJointProbabilityCalibration()
        

    cdef cppclass IOutputWiseProbabilityPredictorMixin:

        # Functions:

        IOutputWiseProbabilityPredictorConfig& useOutputWiseProbabilityPredictor()


    cdef cppclass IMarginalizedProbabilityPredictorMixin:

        # Functions:

        IMarginalizedProbabilityPredictorConfig& useMarginalizedProbabilityPredictor()


    cdef cppclass IAutomaticProbabilityPredictorMixin:
        
        # Functions:

        void useAutomaticProbabilityPredictor()


    cdef cppclass IOutputWiseBinaryPredictorMixin:

        # Functions:

        IOutputWiseBinaryPredictorConfig& useOutputWiseBinaryPredictor()


    cdef cppclass IExampleWiseBinaryPredictorMixin:

        # Functions:

        IExampleWiseBinaryPredictorConfig& useExampleWiseBinaryPredictor()


    cdef cppclass IGfmBinaryPredictorMixin:

        # Functions:

        IGfmBinaryPredictorConfig& useGfmBinaryPredictor()


    cdef cppclass IAutomaticBinaryPredictorMixin:

        # Functions:

        void useAutomaticBinaryPredictor()
