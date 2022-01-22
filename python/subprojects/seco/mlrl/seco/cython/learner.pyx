"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.seco.cython.head_type cimport SingleLabelHeadConfig, PartialHeadConfig
from mlrl.seco.cython.heuristic cimport AccuracyConfig, FMeasureConfig, LaplaceConfig, MEstimateConfig, \
    PrecisionConfig, RecallConfig, WraConfig
from mlrl.seco.cython.lift_function cimport PeakLiftFunctionConfig
from mlrl.seco.cython.predictor cimport LabelWiseClassificationPredictorConfig
from mlrl.seco.cython.stopping_criterion cimport CoverageStoppingCriterionConfig

from libcpp.memory cimport make_unique
from libcpp.utility cimport move


cdef class SeCoRuleLearnerConfig(RuleLearnerConfig):
    """
    A wrapper for the pure virtual C++ class `ISeCoRuleLearner::IConfig`.
    """

    def __cinit__(self):
        self.rule_learner_config_ptr = createSeCoRuleLearnerConfig()

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self):
        return self.rule_learner_config_ptr.get()

    def use_no_coverage_stopping_criterion(self):
        """
        Configures the rule learner to not use any stopping criterion that stops the induction of rules as soon as the
        sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoCoverageStoppingCriterion()

    def use_coverage_stopping_criterion(self) -> CoverageStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that stops the induction of rules as soon as the sum of
        the weights of the uncovered labels is smaller or equal to a certain threshold.

        :return: A `CoverageStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ICoverageStoppingCriterionConfig* config_ptr = &rule_learner_config_ptr.useCoverageStoppingCriterion()
        cdef CoverageStoppingCriterionConfig config = CoverageStoppingCriterionConfig.__new__(CoverageStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_single_label_heads(self) -> SingleLabelHeadConfig:
        """
        Configures the rule learner to induce rules with single-label heads that predict for a single label.

        :return: A `SingleLabelHeadConfig` that allows further configuration of the rule heads
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ISingleLabelHeadConfig* config_ptr = &rule_learner_config_ptr.useSingleLabelHeads()
        cdef SingleLabelHeadConfig config = SingleLabelHeadConfig.__new__(SingleLabelHeadConfig)
        config.config_ptr = config_ptr
        return config

    def use_partial_heads(self) -> PartialHeadConfig:
        """
        Configures the rule learner to induce rules with partial heads that predict for a subset of the available
        labels.

        :return: A `PartialHeadConfig` that allows further configuration of the rule heads
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IPartialHeadConfig* config_ptr = &rule_learner_config_ptr.usePartialHeads()
        cdef PartialHeadConfig config = PartialHeadConfig.__new__(PartialHeadConfig)
        config.config_ptr = config_ptr
        return config

    def use_accuracy_heuristic(self) -> AccuracyConfig:
        """
        Configures the rule learner to use the "Accuracy" heuristic for learning rules.

        :return: An `AccuracyConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IAccuracyConfig* config_ptr = &rule_learner_config_ptr.useAccuracyHeuristic()
        cdef AccuracyConfig config = AccuracyConfig.__new__(AccuracyConfig)
        config.config_ptr = config_ptr
        return config

    def use_f_measure_heuristic(self) -> FMeasureConfig:
        """
        Configures the rule learner to use the "F-Measure" heuristic for learning rules.

        :return: A `FMeasureConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IFMeasureConfig* config_ptr = &rule_learner_config_ptr.useFMeasureHeuristic()
        cdef FMeasureConfig config = FMeasureConfig.__new__(FMeasureConfig)
        config.config_ptr = config_ptr
        return config

    def use_laplace_heuristic(self) -> LaplaceConfig:
        """
        Configures the rule learner to use the "Laplace" heuristic for learning rules.

        :return: A `LaplaceConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILaplaceConfig* config_ptr = &rule_learner_config_ptr.useLaplaceHeuristic()
        cdef LaplaceConfig config = LaplaceConfig.__new__(LaplaceConfig)
        config.config_ptr = config_ptr
        return config

    def use_m_estimate_heuristic(self) -> MEstimateConfig:
        """
        Configures the rule learner to use the "M-Estimate" heuristic for learning rules.

        :return: A `MEstimateConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IMEstimateConfig* config_ptr = &rule_learner_config_ptr.useMEstimateHeuristic()
        cdef MEstimateConfig config = MEstimateConfig.__new__(MEstimateConfig)
        config.config_ptr = config_ptr
        return config

    def use_precision_heuristic(self) -> PrecisionConfig:
        """
        Configures the rule learner to use the "Precision" heuristic for learning rules.

        :return: A `PrecisionConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IPrecisionConfig* config_ptr = &rule_learner_config_ptr.usePrecisionHeuristic()
        cdef PrecisionConfig config = PrecisionConfig.__new__(PrecisionConfig)
        config.config_ptr = config_ptr
        return config

    def use_recall_heuristic(self) -> RecallConfig:
        """
        Configures the rule learner to use the "Recall" heuristic for learning rules.

        :return: A `RecallConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IRecallConfig* config_ptr = &rule_learner_config_ptr.useRecallHeuristic()
        cdef RecallConfig config = RecallConfig.__new__(RecallConfig)
        config.config_ptr = config_ptr
        return config

    def use_wra__heuristic(self) -> WraConfig:
        """
        Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for learning rules.

        :return: A `WraConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IWraConfig* config_ptr = &rule_learner_config_ptr.useWraHeuristic()
        cdef WraConfig config = WraConfig.__new__(WraConfig)
        config.config_ptr = config_ptr
        return config

    def use_accuracy_pruning_heuristic(self) -> AccuracyConfig:
        """
        Configures the rule learner to use the "Accuracy" heuristic for pruning rules.

        :return: An `AccuracyConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IAccuracyConfig* config_ptr = &rule_learner_config_ptr.useAccuracyPruningHeuristic()
        cdef AccuracyConfig config = AccuracyConfig.__new__(AccuracyConfig)
        config.config_ptr = config_ptr
        return config

    def use_f_measure_pruning_heuristic(self) -> FMeasureConfig:
        """
        Configures the rule learner to use the "F-Measure" heuristic for pruning rules.

        :return: A `FMeasureConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IFMeasureConfig* config_ptr = &rule_learner_config_ptr.useFMeasurePruningHeuristic()
        cdef FMeasureConfig config = FMeasureConfig.__new__(FMeasureConfig)
        config.config_ptr = config_ptr
        return config

    def use_laplace_pruning_heuristic(self) -> LaplaceConfig:
        """
        Configures the rule learner to use the "Laplace" heuristic for pruning rules.

        :return: A `LaplaceConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILaplaceConfig* config_ptr = &rule_learner_config_ptr.useLaplacePruningHeuristic()
        cdef LaplaceConfig config = LaplaceConfig.__new__(LaplaceConfig)
        config.config_ptr = config_ptr
        return config

    def use_m_estimate_pruning_heuristic(self) -> MEstimateConfig:
        """
        Configures the rule learner to use the "M-Estimate" heuristic for pruning rules.

        :return: A `MEstimateConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IMEstimateConfig* config_ptr = &rule_learner_config_ptr.useMEstimatePruningHeuristic()
        cdef MEstimateConfig config = MEstimateConfig.__new__(MEstimateConfig)
        config.config_ptr = config_ptr
        return config

    def use_precision_pruning_heuristic(self) -> PrecisionConfig:
        """
        Configures the rule learner to use the "Precision" heuristic for pruning rules.

        :return: A `PrecisionConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IPrecisionConfig* config_ptr = &rule_learner_config_ptr.usePrecisionPruningHeuristic()
        cdef PrecisionConfig config = PrecisionConfig.__new__(PrecisionConfig)
        config.config_ptr = config_ptr
        return config

    def use_recall_pruning_heuristic(self) -> RecallConfig:
        """
        Configures the rule learner to use the "Recall" heuristic for pruning rules.

        :return: A `RecallConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IRecallConfig* config_ptr = &rule_learner_config_ptr.useRecallPruningHeuristic()
        cdef RecallConfig config = RecallConfig.__new__(RecallConfig)
        config.config_ptr = config_ptr
        return config

    def use_wra_pruning_heuristic(self) -> WraConfig:
        """
        Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for pruning rules.

        :return: A `WraConfig` that allows further configuration of the heuristic
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IWraConfig* config_ptr = &rule_learner_config_ptr.useWraPruningHeuristic()
        cdef WraConfig config = WraConfig.__new__(WraConfig)
        config.config_ptr = config_ptr
        return config

    def use_peak_lift_function(self) -> PeakLiftFunctionConfig:
        """
        Configures the rule learner to use a lift function that monotonously increases until a certain number of labels,
        where the maximum lift is reached, and monotonously decreases afterwards.

        :return: A `PeakLiftFunctionConfig` that allows further configuration of the lift function
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IPeakLiftFunctionConfig* config_ptr = &rule_learner_config_ptr.usePeakLiftFunction()
        cdef PeakLiftFunctionConfig config = PeakLiftFunctionConfig.__new__(PeakLiftFunctionConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_classification_predictor(self) -> LabelWiseClassificationPredictorConfig:
        """
        Configures the rule learner to use predictor for predicting whether individual labels of given query examples
        are relevant or irrelevant by processing rules of an existing rule-based model in the order they have been
        learned. If a rule covers an example, its prediction is applied to each label individually.

        :return: A `LabelWiseClassificationPredictorConfig` that allows further configuration of the predictor for
                 predicting whether individual labels of given query examples are relevant or irrelevant
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelWiseClassificationPredictorConfig* config_ptr = &rule_learner_config_ptr.useLabelWiseClassificationPredictor()
        cdef LabelWiseClassificationPredictorConfig config = LabelWiseClassificationPredictorConfig.__new__(LabelWiseClassificationPredictorConfig)
        config.config_ptr = config_ptr
        return config


cdef class SeCoRuleLearner(RuleLearner):
    """
    A wrapper for the pure virtual C++ class `ISeCoRuleLearner`.
    """

    def __cinit__(self, SeCoRuleLearnerConfig config not None):
        """
        :param config: The configuration that should be used by the rule learner
        """
        self.rule_learner_ptr = createSeCoRuleLearner(move(config.rule_learner_config_ptr))

    cdef IRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
