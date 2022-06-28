from mlrl.common.cython.learner cimport IRuleLearner, RuleLearner, IBeamSearchTopDownMixin, IFeatureBinningMixin, \
    ILabelSamplingMixin, IInstanceSamplingMixin, IFeatureSamplingMixin, IPartitionSamplingMixin, IPruningMixin, \
    IMultiThreadingMixin, ISizeStoppingCriterionMixin, ITimeStoppingCriterionMixin, IMeasureStoppingCriterionMixin
from mlrl.seco.cython.learner cimport ISeCoRuleLearnerConfig, SeCoRuleLearnerConfig

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner_seco.hpp" namespace "seco" nogil:

    cdef cppclass IMultiLabelSeCoRuleLearnerConfig"seco::IMultiLabelSeCoRuleLearner::IConfig"(
            ISeCoRuleLearnerConfig,
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
        pass


    cdef cppclass IMultiLabelSeCoRuleLearner(IRuleLearner):
        pass


    unique_ptr[IMultiLabelSeCoRuleLearnerConfig] createMultiLabelSeCoRuleLearnerConfig()


    unique_ptr[IMultiLabelSeCoRuleLearner] createMultiLabelSeCoRuleLearner(
        unique_ptr[IMultiLabelSeCoRuleLearnerConfig] configPtr)


cdef class MultiLabelSeCoRuleLearnerConfig(SeCoRuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearnerConfig] rule_learner_config_ptr


cdef class MultiLabelSeCoRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[IMultiLabelSeCoRuleLearner] rule_learner_ptr
