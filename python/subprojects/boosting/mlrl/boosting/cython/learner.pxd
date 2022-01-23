from mlrl.common.cython.learner cimport IRuleLearner, IRuleLearnerConfig, RuleLearner, RuleLearnerConfig
from mlrl.boosting.cython.label_binning cimport IEqualWidthLabelBinningConfig
from mlrl.boosting.cython.loss cimport IExampleWiseLogisticLossConfig, ILabelWiseLogisticLossConfig, \
    ILabelWiseSquaredErrorLossConfig, ILabelWiseSquaredHingeLossConfig
from mlrl.boosting.cython.post_processor cimport IConstantShrinkageConfig
from mlrl.boosting.cython.predictor cimport IExampleWiseClassificationPredictorConfig, \
    ILabelWiseClassificationPredictorConfig, ILabelWiseRegressionPredictorConfig, ILabelWiseProbabilityPredictorConfig

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass IBoostingRuleLearnerConfig"boosting::IBoostingRuleLearner::IConfig"(IRuleLearnerConfig):

        # Functions:

        void useAutomaticFeatureBinning()

        IConstantShrinkageConfig& useConstantShrinkagePostProcessor()

        void useAutomaticParallelRuleRefinement()

        void useAutomaticParallelStatisticUpdate()

        void useSingleLabelHeads()

        void useCompleteHeads()

        IExampleWiseLogisticLossConfig& useExampleWiseLogisticLoss()

        ILabelWiseLogisticLossConfig& useLabelWiseLogisticLoss()

        ILabelWiseSquaredErrorLossConfig& useLabelWiseSquaredErrorLoss()

        ILabelWiseSquaredHingeLossConfig& useLabelWiseSquaredHingeLoss()

        void useNoLabelBinning()

        IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning()

        IExampleWiseClassificationPredictorConfig& useExampleWiseClassificationPredictor()

        ILabelWiseClassificationPredictorConfig& useLabelWiseClassificationPredictor()

        ILabelWiseRegressionPredictorConfig& useLabelWiseRegressionPredictor()

        ILabelWiseProbabilityPredictorConfig& useLabelWiseProbabilityPredictor()


    cdef cppclass IBoostingRuleLearner(IRuleLearner):
        pass


    unique_ptr[IBoostingRuleLearnerConfig] createBoostingRuleLearnerConfig()


    unique_ptr[IBoostingRuleLearner] createBoostingRuleLearner(unique_ptr[IBoostingRuleLearnerConfig] configPtr)


cdef class BoostingRuleLearnerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoostingRuleLearnerConfig] rule_learner_config_ptr


cdef class BoostingRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[IBoostingRuleLearner] rule_learner_ptr
