
from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, IBeamSearchTopDownMixin, \
    IEqualWidthFeatureBinningMixin, IEqualFrequencyFeatureBinningMixin, ILabelSamplingWithoutReplacementMixin, \
    IInstanceSamplingWithoutReplacementMixin, IInstanceSamplingWithReplacementMixin, \
    ILabelWiseStratifiedInstanceSamplingMixin, IExampleWiseStratifiedInstanceSamplingMixin, \
    IFeatureSamplingWithoutReplacementMixin, IPartitionSamplingMixin, IRulePruningMixin, IMultiThreadingMixin, \
    ISizeStoppingCriterionMixin, ITimeStoppingCriterionMixin, IPrePruningMixin, IPostPruningMixin, \
    ISequentialPostOptimizationMixin
from mlrl.boosting.cython.learner cimport IBoostingRuleLearnerConfig, BoostingRuleLearnerConfig, IShrinkageMixin, \
    IL1RegularizationMixin, IL2RegularizationMixin, INoDefaultRuleMixin, IDynamicPartialHeadMixin, \
    IFixedPartialHeadMixin, ISingleLabelHeadMixin, ISparseStatisticsMixin, IExampleWiseLogisticLossMixin, \
    IExampleWiseSquaredErrorLossMixin, IExampleWiseSquaredHingeLossMixin, ILabelWiseSquaredErrorLossMixin, \
    ILabelWiseSquaredHingeLossMixin, ILabelBinningMixin, IExampleWiseBinaryPredictorMixin, IGfmBinaryPredictorMixin, \
    IMarginalizedProbabilityPredictorMixin, DdotFunction, DspmvFunction, DsysvFunction

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/learner_boomer.hpp" namespace "boosting" nogil:

    cdef cppclass IBoomerConfig"boosting::IBoomer::IConfig"(IBoostingRuleLearnerConfig,
                                                            IShrinkageMixin,
                                                            IL1RegularizationMixin,
                                                            IL2RegularizationMixin,
                                                            INoDefaultRuleMixin,
                                                            IDynamicPartialHeadMixin,
                                                            IFixedPartialHeadMixin,
                                                            ISingleLabelHeadMixin,
                                                            ISparseStatisticsMixin,
                                                            IExampleWiseLogisticLossMixin,
                                                            IExampleWiseSquaredErrorLossMixin,
                                                            IExampleWiseSquaredHingeLossMixin,
                                                            ILabelWiseSquaredErrorLossMixin,
                                                            ILabelWiseSquaredHingeLossMixin,
                                                            ILabelBinningMixin,
                                                            IExampleWiseBinaryPredictorMixin,
                                                            IGfmBinaryPredictorMixin,
                                                            IMarginalizedProbabilityPredictorMixin,
                                                            IBeamSearchTopDownMixin,
                                                            IEqualWidthFeatureBinningMixin,
                                                            IEqualFrequencyFeatureBinningMixin,
                                                            ILabelSamplingWithoutReplacementMixin,
                                                            IInstanceSamplingWithoutReplacementMixin,
                                                            IInstanceSamplingWithReplacementMixin,
                                                            ILabelWiseStratifiedInstanceSamplingMixin,
                                                            IExampleWiseStratifiedInstanceSamplingMixin,
                                                            IFeatureSamplingWithoutReplacementMixin,
                                                            IPartitionSamplingMixin,
                                                            IRulePruningMixin,
                                                            IMultiThreadingMixin,
                                                            ISizeStoppingCriterionMixin,
                                                            ITimeStoppingCriterionMixin,
                                                            IPrePruningMixin,
                                                            IPostPruningMixin,
                                                            ISequentialPostOptimizationMixin):

        # Functions:

        void useAutomaticDefaultRule()

        void useAutomaticPartitionSampling()

        void useAutomaticFeatureBinning()

        void useAutomaticParallelRuleRefinement()

        void useAutomaticParallelStatisticUpdate()

        void useAutomaticHeads()

        void useAutomaticStatistics()

        void useAutomaticLabelBinning()

        void useAutomaticProbabilityPredictor()


    cdef cppclass IBoomer(IRuleLearner):
        pass


    unique_ptr[IBoomerConfig] createBoomerConfig()


    unique_ptr[IBoomer] createBoomer(unique_ptr[IBoomerConfig] configPtr, DdotFunction ddotFunction,
                                     DspmvFunction dspmvFunction, DsysvFunction dsysvFunction)


cdef class BoomerConfig(BoostingRuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoomerConfig] rule_learner_config_ptr


cdef class Boomer(RuleLearner):

    # Attributes:

    cdef unique_ptr[IBoomer] rule_learner_ptr
