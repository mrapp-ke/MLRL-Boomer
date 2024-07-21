/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_binning_equal_frequency.hpp"
#include "mlrl/common/input/feature_binning_equal_width.hpp"
#include "mlrl/common/input/feature_binning_no.hpp"
#include "mlrl/common/multi_threading/multi_threading_manual.hpp"
#include "mlrl/common/multi_threading/multi_threading_no.hpp"
#include "mlrl/common/post_optimization/post_optimization_no.hpp"
#include "mlrl/common/post_optimization/post_optimization_phase_list.hpp"
#include "mlrl/common/post_optimization/post_optimization_sequential.hpp"
#include "mlrl/common/post_optimization/post_optimization_unused_rule_removal.hpp"
#include "mlrl/common/post_processing/post_processor_no.hpp"
#include "mlrl/common/prediction/output_space_info.hpp"
#include "mlrl/common/prediction/prediction_matrix_dense.hpp"
#include "mlrl/common/prediction/prediction_matrix_sparse_binary.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/prediction/probability_calibration_no.hpp"
#include "mlrl/common/rule_induction/rule_induction_top_down_beam_search.hpp"
#include "mlrl/common/rule_induction/rule_induction_top_down_greedy.hpp"
#include "mlrl/common/rule_model_assemblage/default_rule.hpp"
#include "mlrl/common/rule_model_assemblage/rule_model_assemblage.hpp"
#include "mlrl/common/rule_model_assemblage/rule_model_assemblage_sequential.hpp"
#include "mlrl/common/rule_pruning/rule_pruning_irep.hpp"
#include "mlrl/common/rule_pruning/rule_pruning_no.hpp"
#include "mlrl/common/sampling/feature_sampling_no.hpp"
#include "mlrl/common/sampling/feature_sampling_without_replacement.hpp"
#include "mlrl/common/sampling/instance_sampling_no.hpp"
#include "mlrl/common/sampling/instance_sampling_with_replacement.hpp"
#include "mlrl/common/sampling/instance_sampling_without_replacement.hpp"
#include "mlrl/common/sampling/output_sampling_no.hpp"
#include "mlrl/common/sampling/output_sampling_round_robin.hpp"
#include "mlrl/common/sampling/output_sampling_without_replacement.hpp"
#include "mlrl/common/sampling/partition_sampling_bi_random.hpp"
#include "mlrl/common/sampling/partition_sampling_no.hpp"
#include "mlrl/common/stopping/global_pruning_no.hpp"
#include "mlrl/common/stopping/global_pruning_post.hpp"
#include "mlrl/common/stopping/global_pruning_pre.hpp"
#include "mlrl/common/stopping/stopping_criterion_list.hpp"
#include "mlrl/common/stopping/stopping_criterion_no.hpp"
#include "mlrl/common/stopping/stopping_criterion_size.hpp"
#include "mlrl/common/stopping/stopping_criterion_time.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>
#include <utility>

/**
 * Defines an interface for all classes that provide access to the results of fitting a rule learner to training data.
 * It incorporates the model that has been trained, as well as additional information that is necessary for obtaining
 * predictions for unseen data.
 */
class MLRLCOMMON_API ITrainingResult {
    public:

        virtual ~ITrainingResult() {}

        /**
         * Returns the number of outputs for which a model has been trained.
         *
         * @return The number of outputs
         */
        virtual uint32 getNumOutputs() const = 0;

        /**
         * Returns the model that has been trained.
         *
         * @return An unique pointer to an object of type `IRuleModel` that has been trained
         */
        virtual std::unique_ptr<IRuleModel>& getRuleModel() = 0;

        /**
         * Returns the model that has been trained.
         *
         * @return An unique pointer to an object of type `IRuleModel` that has been trained
         */
        virtual const std::unique_ptr<IRuleModel>& getRuleModel() const = 0;

        /**
         * Returns information about the output space that may be used as a basis for making predictions.
         *
         * @return An unique pointer to an object of type `IOutputSpaceInfo` that may be used as a basis for making
         *         predictions
         */
        virtual std::unique_ptr<IOutputSpaceInfo>& getOutputSpaceInfo() = 0;

        /**
         * Returns information about the output space that may be used as a basis for making predictions.
         *
         * @return An unique pointer to an object of type `IOutputSpaceInfo` that may be used as a basis for making
         *         predictions
         */
        virtual const std::unique_ptr<IOutputSpaceInfo>& getOutputSpaceInfo() const = 0;

        /**
         * Returns a model that may be used for the calibration of marginal probabilities.
         *
         * @return An unique pointer to an object of type `IMarginalProbabilityCalibrationModel` that may be used for
         *         the calibration of marginal probabilities
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel>& getMarginalProbabilityCalibrationModel() = 0;

        /**
         * Returns a model that may be used for the calibration of marginal probabilities.
         *
         * @return An unique pointer to an object of type `IMarginalProbabilityCalibrationModel` that may be used for
         *         the calibration of marginal probabilities
         */
        virtual const std::unique_ptr<IMarginalProbabilityCalibrationModel>& getMarginalProbabilityCalibrationModel()
          const = 0;

        /**
         * Returns a model that may be used for the calibration of joint probabilities.
         *
         * @return An unique pointer to an object of type `IJointProbabilityCalibrationModel` that may be used for the
         *         calibration of joint probabilities
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel>& getJointProbabilityCalibrationModel() = 0;

        /**
         * Returns a model that may be used for the calibration of joint probabilities.
         *
         * @return An unique pointer to an object of type `IJointProbabilityCalibrationModel` that may be used for the
         *         calibration of joint probabilities
         */
        virtual const std::unique_ptr<IJointProbabilityCalibrationModel>& getJointProbabilityCalibrationModel()
          const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a rule learner.
 */
class MLRLCOMMON_API IRuleLearnerConfig {
    public:

        virtual ~IRuleLearnerConfig() {}

        /**
         * Returns the definition of the function that should be used for comparing the quality of different rules.
         *
         * @return An object of type `RuleCompareFunction` that defines the function that should be used for comparing
         *         the quality of different rules
         */
        virtual RuleCompareFunction getRuleCompareFunction() const = 0;

        /**
         * Returns a `Property` that allows to access the `IDefaultRuleConfig` that stores the configuration of the
         * default rule.
         *
         * @return A `Property` that allows to access the `IDefaultRuleConfig` that stores the configuration of the
         *         default rule
         */
        virtual Property<IDefaultRuleConfig> getDefaultRuleConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IRuleModelAssemblageConfig` that stores the configuration of
         * the algorithm for the induction of several rules that will be added to a rule-based model.
         *
         * @return A `Property` that allows to access the `IRuleModelAssemblageConfig` that stores the configuration of
         *         the algorithm for the induction of several rules that will be added to a rule-based model
         */
        virtual Property<IRuleModelAssemblageConfig> getRuleModelAssemblageConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IRuleInductionConfig` that stores the configuration of the
         * algorithm for the induction of individual rules.
         *
         * @return A `Property` that allows to access the `IRuleInductionConfig` that stores the configuration of the
         *         algorithm for the induction of individual rules
         */
        virtual Property<IRuleInductionConfig> getRuleInductionConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IFeatureBinningConfig` that stores the configuration of the
         * method for the assignment of numerical feature values to bins.
         *
         * @return A `Property` that allows to access the `IFeatureBinningConfig` that stores the configuration of the
         *         method for the assignment of numerical feature values to bins
         */
        virtual Property<IFeatureBinningConfig> getFeatureBinningConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IOutputSamplingConfig` that stores the configuration of the
         * method for sampling outputs.
         *
         * @return A `Property` that allows to access the `IOutputSamplingConfig` that stores the configuration of the
         *         method for sampling outputs
         */
        virtual Property<IOutputSamplingConfig> getOutputSamplingConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IClassificationInstanceSamplingConfig` that stores the
         * configuration of the method for sampling instances.
         *
         * @return A `Property` that allows to access the `IClassificationInstanceSamplingConfig` that stores the
         * configuration of the method for sampling instances
         */
        virtual Property<IClassificationInstanceSamplingConfig> getClassificationInstanceSamplingConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IFeatureSamplingConfig` that stores the configuration of the
         * method for sampling features.
         *
         * @return A `Property` that allows to access the `IFeatureSamplingConfig` that stores the configuration of the
         *         method for sampling features
         */
        virtual Property<IFeatureSamplingConfig> getFeatureSamplingConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IClassificationPartitionSamplingConfig` that stores the
         * configuration of the method for partitioning the available training examples into a training set and a
         * holdout set.
         *
         * @return A `Property` that allows to access the `IClassificationPartitionSamplingConfig` that stores the
         *         configuration of the method for partitioning the available training examples into a training set and
         *         a holdout set
         */
        virtual Property<IClassificationPartitionSamplingConfig> getClassificationPartitionSamplingConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IRulePruningConfig` that stores the configuration of the
         * method for pruning individual rules.
         *
         * @return A `Property` that allows to access the `IRulePruningConfig` that stores the configuration of the
         *         method for pruning individual rules
         */
        virtual Property<IRulePruningConfig> getRulePruningConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IPostProcessorConfig` that stores the configuration of the
         * method for post-processing the predictions of rules once they have been learned.
         *
         * @return A `Property` that allows to access the `IPostProcessorConfig` that stores the configuration of the
         *         method that post-processes the predictions of rules once they have been learned
         */
        virtual Property<IPostProcessorConfig> getPostProcessorConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IMultiThreadingConfig` that stores the configuration of the
         * multi-threading behavior that is used for the parallel refinement of rules.
         *
         * @return A `Property` that allows to access the `IMultiThreadingConfig` that stores the configuration of the
         *         multi-threading behavior that is used for the parallel refinement of rules
         */
        virtual Property<IMultiThreadingConfig> getParallelRuleRefinementConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IMultiThreadingConfig` that stores the configuration of the
         * multi-threading behavior that is used for the parallel update of statistics.
         *
         * @return A `Property` that allows to access the `IMultiThreadingConfig` that stores the configuration of the
         *         multi-threading behavior that is used for the parallel update of statistics
         */
        virtual Property<IMultiThreadingConfig> getParallelStatisticUpdateConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IMultiThreadingConfig` that stores the configuration of the
         * multi-threading behavior that is used to predict for several query examples in parallel.
         *
         * @return A `Property` that allows to access the `IMultiThreadingConfig` that stores the configuration of the
         *         multi-threading behavior that is used to predict for several query examples in parallel
         */
        virtual Property<IMultiThreadingConfig> getParallelPredictionConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IStoppingCriterionConfig` that stores the configuration of
         * the stopping criterion that ensures that the number of rules does not exceed a certain maximum.
         *
         * @return A `Property` that allows to access the `IStoppingCriterionConfig` that stores the configuration of
         *         the stopping criterion that ensures that the number of rules does not exceed a certain maximum
         */
        virtual Property<IStoppingCriterionConfig> getSizeStoppingCriterionConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IStoppingCriterionConfig` that stores the configuration of
         * the stopping criterion that ensures that a certain time limit is not exceeded.
         *
         * @return A `Property` that allows to access the `IStoppingCriterionConfig` that stores the configuration of
         *         the stopping criterion that ensures that a certain time limit is not exceeded
         */
        virtual Property<IStoppingCriterionConfig> getTimeStoppingCriterionConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IGlobalPruningConfig` that stores the configuration of the
         * stopping criterion that allows to decide how many rules should be included in a model, such that its
         * performance is optimized globally.
         *
         * @return A `Property` that allows to access the `IGlobalPruningConfig` that stores the configuration of the
         *         stopping criterion that allows to decide how many rules should be included in a model
         */
        virtual Property<IGlobalPruningConfig> getGlobalPruningConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IPostOptimizationPhaseConfig` that stores the configuration
         * of the post-optimization method that optimizes each rule in a model by relearning it in the context of the
         * other rules.
         *
         * @return A `Property` that allows to access the `IPostOptimizationPhaseConfig` that stores the configuration
         *         of the post-optimization method that optimizes each rule in a model by relearning it in the context
         *         of the other rules
         */
        virtual Property<IPostOptimizationPhaseConfig> getSequentialPostOptimizationConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IPostOptimizationPhaseConfig` that stores the configuration
         * of the post-optimization method that removes unused rules from a model.
         *
         * @return A `Property` that allows to access the `IPostOptimizationPhaseConfig` that stores the configuration
         *         of the post-optimization method that removes unused rules from a model
         */
        virtual Property<IPostOptimizationPhaseConfig> getUnusedRuleRemovalConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IMarginalProbabilityCalibratorConfig` that stores the
         * configuration of the calibrator that allows to fit a model for the calibration of marginal probabilities.
         *
         * @return A `Property` that allows to access the `IMarginalProbabilityCalibratorConfig` that stores the
         *         configuration of the calibrator that allows to fit a model for the calibration of marginal
         *         probabilities
         */
        virtual Property<IMarginalProbabilityCalibratorConfig> getMarginalProbabilityCalibratorConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IJointProbabilityCalibratorConfig` that stores the
         * configuration of the calibrator that allows to fit a model for the calibration of joint probabilities.
         *
         * @return A `Property` that allows to access the `IJointProbabilityCalibratorConfig` that stores the
         *         configuration of the calibrator that allows to fit a model for the calibration of joint probabilities
         */
        virtual Property<IJointProbabilityCalibratorConfig> getJointProbabilityCalibratorConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IScorePredictorConfig` that stores the configuration of the
         * predictor that allows to predict scores.
         *
         * @return A `Property` that allows to access the `IScorePredictorConfig` that stores the configuration of the
         *         predictor that allows to predict scores
         */
        virtual Property<IScorePredictorConfig> getScorePredictorConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IProbabilityPredictorConfig` that stores the configuration of
         * the predictor that allows to predict probability estimates.
         *
         * @return A `Property` that allows to access the `IProbabilityPredictorConfig` that stores the configuration of
         *         the predictor that allows to predict probability estimates
         */
        virtual Property<IProbabilityPredictorConfig> getProbabilityPredictorConfig() = 0;

        /**
         * Returns a `Property` that allows to access the `IBinaryPredictorConfig` that stores the configuration of the
         * predictor that allows to predict binary labels.
         *
         * @return A `Property` that allows to access the `IBinaryPredictorConfig` that stores the configuration of the
         *         predictor that allows to predict binary labels
         */
        virtual Property<IBinaryPredictorConfig> getBinaryPredictorConfig() = 0;
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use an algorithm that sequentially
 * induces several rules.
 */
class ISequentialRuleModelAssemblageMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~ISequentialRuleModelAssemblageMixin() override {}

        /**
         * Configures the rule learner to use an algorithm that sequentially induces several rules, optionally starting
         * with a default rule, that are added to a rule-based model.
         */
        virtual void useSequentialRuleModelAssemblage() {
            Property<IRuleModelAssemblageConfig> property = this->getRuleModelAssemblageConfig();
            property.set(std::make_unique<SequentialRuleModelAssemblageConfig>(this->getDefaultRuleConfig()));
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to induce a default rule.
 */
class IDefaultRuleMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IDefaultRuleMixin() override {}

        /**
         * Configures the rule learner to induce a default rule.
         */
        virtual void useDefaultRule() {
            Property<IDefaultRuleConfig> property = this->getDefaultRuleConfig();
            property.set(std::make_unique<DefaultRuleConfig>(true));
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use a greedy top-down search for the
 * induction of individual rules.
 */
class MLRLCOMMON_API IGreedyTopDownRuleInductionMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IGreedyTopDownRuleInductionMixin() override {}

        /**
         * Configures the rule learner to use a greedy top-down search for the induction of individual rules.
         *
         * @return A reference to an object of type `IGreedyTopDownRuleInductionConfig` that allows further
         *         configuration of the algorithm for the induction of individual rules
         */
        virtual IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction() {
            Property<IRuleInductionConfig> property = this->getRuleInductionConfig();
            std::unique_ptr<GreedyTopDownRuleInductionConfig> ptr = std::make_unique<GreedyTopDownRuleInductionConfig>(
              this->getRuleCompareFunction(), this->getParallelRuleRefinementConfig());
            IGreedyTopDownRuleInductionConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use a top-down beam search.
 */
class MLRLCOMMON_API IBeamSearchTopDownRuleInductionMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IBeamSearchTopDownRuleInductionMixin() override {}

        /**
         * Configures the rule learner to use a top-down beam search for the induction of individual rules.
         *
         * @return A reference to an object of type `IBeamSearchTopDownRuleInduction` that allows further configuration
         *         of the algorithm for the induction of individual rules
         */
        virtual IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction() {
            Property<IRuleInductionConfig> property = this->getRuleInductionConfig();
            std::unique_ptr<BeamSearchTopDownRuleInductionConfig> ptr =
              std::make_unique<BeamSearchTopDownRuleInductionConfig>(this->getRuleCompareFunction(),
                                                                     this->getParallelRuleRefinementConfig());
            IBeamSearchTopDownRuleInductionConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use any post processor.
 */
class MLRLCOMMON_API INoPostProcessorMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoPostProcessorMixin() override {}

        /**
         * Configures the rule learner to not use any post processor.
         */
        virtual void useNoPostProcessor() {
            Property<IPostProcessorConfig> property = this->getPostProcessorConfig();
            property.set(std::make_unique<NoPostProcessorConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use any method for the assignment
 * of numerical features values to bins.
 */
class MLRLCOMMON_API INoFeatureBinningMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoFeatureBinningMixin() override {}

        /**
         * Configures the rule learner to not use any method for the assignment of numerical feature values to bins.
         */
        virtual void useNoFeatureBinning() {
            Property<IFeatureBinningConfig> property = this->getFeatureBinningConfig();
            property.set(std::make_unique<NoFeatureBinningConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use equal-width feature binning.
 */
class MLRLCOMMON_API IEqualWidthFeatureBinningMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IEqualWidthFeatureBinningMixin() override {}

        /**
         * Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
         * each bin contains values from equally sized value ranges.
         *
         * @return A reference to an object of type `IEqualWidthFeatureBinningConfig` that allows further configuration
         *         of the method for the assignment of numerical feature values to bins
         */
        virtual IEqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning() {
            Property<IFeatureBinningConfig> property = this->getFeatureBinningConfig();
            std::unique_ptr<EqualWidthFeatureBinningConfig> ptr = std::make_unique<EqualWidthFeatureBinningConfig>();
            IEqualWidthFeatureBinningConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use equal-frequency feature binning.
 */
class MLRLCOMMON_API IEqualFrequencyFeatureBinningMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IEqualFrequencyFeatureBinningMixin() override {}

        /**
         * Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
         * each bin contains approximately the same number of values.
         *
         * @return A reference to an object of type `IEqualFrequencyFeatureBinningConfig` that allows further
         *         configuration of the method for the assignment of numerical feature values to bins
         */
        virtual IEqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning() {
            Property<IFeatureBinningConfig> property = this->getFeatureBinningConfig();
            std::unique_ptr<EqualFrequencyFeatureBinningConfig> ptr =
              std::make_unique<EqualFrequencyFeatureBinningConfig>();
            IEqualFrequencyFeatureBinningConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use output sampling.
 */
class MLRLCOMMON_API INoOutputSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoOutputSamplingMixin() override {}

        /**
         * Configures the rule learner to not sample from the available outputs whenever a new rule should be learned.
         */
        virtual void useNoOutputSampling() {
            Property<IOutputSamplingConfig> property = this->getOutputSamplingConfig();
            property.set(std::make_unique<NoOutputSamplingConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use output sampling without
 * replacement.
 */
class MLRLCOMMON_API IOutputSamplingWithoutReplacementMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IOutputSamplingWithoutReplacementMixin() override {}

        /**
         * Configures the rule learner to sample from the available outputs with replacement whenever a new rule should
         * be learned.
         *
         * @return A reference to an object of type `IOutputSamplingWithoutReplacementConfig` that allows further
         *         configuration of the sampling method
         */
        virtual IOutputSamplingWithoutReplacementConfig& useOutputSamplingWithoutReplacement() {
            Property<IOutputSamplingConfig> property = this->getOutputSamplingConfig();
            std::unique_ptr<OutputSamplingWithoutReplacementConfig> ptr =
              std::make_unique<OutputSamplingWithoutReplacementConfig>();
            IOutputSamplingWithoutReplacementConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to sample one output at a time in a
 * round-robin fashion.
 */
class MLRLCOMMON_API IRoundRobinOutputSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IRoundRobinOutputSamplingMixin() override {}

        /**
         * Configures the rule learner to sample one output at a time in a round-robin fashion whenever a new rule
         * should be learned.
         */
        virtual void useRoundRobinOutputSampling() {
            Property<IOutputSamplingConfig> property = this->getOutputSamplingConfig();
            property.set(std::make_unique<RoundRobinOutputSamplingConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use instance sampling.
 */
class MLRLCOMMON_API INoInstanceSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoInstanceSamplingMixin() override {}

        /**
         * Configures the rule learner to not sample from the available training examples whenever a new rule should be
         * learned.
         */
        virtual void useNoInstanceSampling() {
            Property<IClassificationInstanceSamplingConfig> property = this->getClassificationInstanceSamplingConfig();
            property.set(std::make_unique<NoInstanceSamplingConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use instance sampling with
 * replacement.
 */
class MLRLCOMMON_API IInstanceSamplingWithReplacementMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IInstanceSamplingWithReplacementMixin() override {}

        /**
         * Configures the rule learner to sample from the available training examples with replacement whenever a new
         * rule should be learned.
         *
         * @return A reference to an object of type `IInstanceSamplingWithReplacementConfig` that allows further
         *         configuration of the method for sampling instances
         */
        virtual IInstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement() {
            Property<IClassificationInstanceSamplingConfig> property = this->getClassificationInstanceSamplingConfig();
            std::unique_ptr<InstanceSamplingWithReplacementConfig> ptr =
              std::make_unique<InstanceSamplingWithReplacementConfig>();
            IInstanceSamplingWithReplacementConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use instance sampling without
 * replacement.
 */
class MLRLCOMMON_API IInstanceSamplingWithoutReplacementMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IInstanceSamplingWithoutReplacementMixin() override {}

        /**
         * Configures the rule learner to sample from the available training examples without replacement whenever a new
         * rule should be learned.
         *
         * @return A reference to an object of type `IInstanceSamplingWithoutReplacementConfig` that allows further
         *         configuration of the method for sampling instances
         */
        virtual IInstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement() {
            Property<IClassificationInstanceSamplingConfig> property = this->getClassificationInstanceSamplingConfig();
            std::unique_ptr<InstanceSamplingWithoutReplacementConfig> ptr =
              std::make_unique<InstanceSamplingWithoutReplacementConfig>();
            IInstanceSamplingWithoutReplacementConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use feature sampling.
 */
class MLRLCOMMON_API INoFeatureSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoFeatureSamplingMixin() override {}

        /**
         * Configures the rule learner to not sample from the available features whenever a rule should be refined.
         */
        virtual void useNoFeatureSampling() {
            Property<IFeatureSamplingConfig> property = this->getFeatureSamplingConfig();
            property.set(std::make_unique<NoFeatureSamplingConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use feature sampling without
 * replacement.
 */
class MLRLCOMMON_API IFeatureSamplingWithoutReplacementMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IFeatureSamplingWithoutReplacementMixin() override {}

        /**
         * Configures the rule learner to sample from the available features with replacement whenever a rule should be
         * refined.
         *
         * @return A reference to an object of type `IFeatureSamplingWithoutReplacementConfig` that allows further
         *         configuration of the method for sampling features
         */
        virtual IFeatureSamplingWithoutReplacementConfig& useFeatureSamplingWithoutReplacement() {
            Property<IFeatureSamplingConfig> property = this->getFeatureSamplingConfig();
            std::unique_ptr<FeatureSamplingWithoutReplacementConfig> ptr =
              std::make_unique<FeatureSamplingWithoutReplacementConfig>();
            IFeatureSamplingWithoutReplacementConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not partition the available training
 * examples into a training set and a holdout set.
 */
class MLRLCOMMON_API INoPartitionSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoPartitionSamplingMixin() override {}

        /**
         * Configures the rule learner to not partition the available training examples into a training set and a
         * holdout set.
         */
        virtual void useNoPartitionSampling() {
            Property<IClassificationPartitionSamplingConfig> property =
              this->getClassificationPartitionSamplingConfig();
            property.set(std::make_unique<NoPartitionSamplingConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to partition the available training
 * example into a training set and a holdout set by randomly splitting the training examples into two mutually exclusive
 * sets.
 */
class MLRLCOMMON_API IRandomBiPartitionSamplingMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IRandomBiPartitionSamplingMixin() override {}

        /**
         * Configures the rule learner to partition the available training examples into a training set and a
         * holdout set by randomly splitting the training examples into two mutually exclusive sets.
         *
         * @return A reference to an object of type `IRandomBiPartitionSamplingConfig` that allows further configuration
         *         of the method for partitioning the available training examples into a training set and a holdout set
         */
        virtual IRandomBiPartitionSamplingConfig& useRandomBiPartitionSampling() {
            Property<IClassificationPartitionSamplingConfig> property =
              this->getClassificationPartitionSamplingConfig();
            std::unique_ptr<RandomBiPartitionSamplingConfig> ptr = std::make_unique<RandomBiPartitionSamplingConfig>();
            IRandomBiPartitionSamplingConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not prune individual rules.
 */
class MLRLCOMMON_API INoRulePruningMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoRulePruningMixin() override {}

        /**
         * Configures the rule learner to not prune individual rules.
         */
        virtual void useNoRulePruning() {
            Property<IRulePruningConfig> property = this->getRulePruningConfig();
            property.set(std::make_unique<NoRulePruningConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to prune individual rules by following
 * the principles of "incremental reduced error pruning" (IREP).
 */
class MLRLCOMMON_API IIrepRulePruningMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IIrepRulePruningMixin() override {}

        /**
         * Configures the rule learner to prune individual rules by following the principles of "incremental reduced
         * error pruning" (IREP).
         */
        virtual void useIrepRulePruning() {
            Property<IRulePruningConfig> property = this->getRulePruningConfig();
            property.set(std::make_unique<IrepConfig>(this->getRuleCompareFunction()));
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use any multi-threading for the
 * parallel refinement of rules.
 */
class MLRLCOMMON_API INoParallelRuleRefinementMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoParallelRuleRefinementMixin() override {}

        /**
         * Configures the rule learner to not use any multi-threading for the parallel refinement of rules.
         */
        virtual void useNoParallelRuleRefinement() {
            Property<IMultiThreadingConfig> property = this->getParallelRuleRefinementConfig();
            property.set(std::make_unique<NoMultiThreadingConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use multi-threading for the parallel
 * refinement of rules.
 */
class MLRLCOMMON_API IParallelRuleRefinementMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IParallelRuleRefinementMixin() override {}

        /**
         * Configures the rule learner to use multi-threading for the parallel refinement of rules.
         *
         * @return A reference to an object of type `IManualMultiThreadingConfig` that allows further configuration of
         *         the multi-threading behavior
         */
        virtual IManualMultiThreadingConfig& useParallelRuleRefinement() {
            Property<IMultiThreadingConfig> property = this->getParallelRuleRefinementConfig();
            std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
            IManualMultiThreadingConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use any multi-threading for the
 * parallel update of statistics.
 */
class MLRLCOMMON_API INoParallelStatisticUpdateMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoParallelStatisticUpdateMixin() override {}

        /**
         * Configures the rule learner to not use any multi-threading for the parallel update of statistics.
         */
        virtual void useNoParallelStatisticUpdate() {
            Property<IMultiThreadingConfig> property = this->getParallelStatisticUpdateConfig();
            property.set(std::make_unique<NoMultiThreadingConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use multi-threading for the parallel
 * update of statistics.
 */
class MLRLCOMMON_API IParallelStatisticUpdateMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IParallelStatisticUpdateMixin() override {}

        /**
         * Configures the rule learner to use multi-threading for the parallel update of statistics.
         *
         * @return A reference to an object of type `IManualMultiThreadingConfig` that allows further configuration of
         *         the multi-threading behavior
         */
        virtual IManualMultiThreadingConfig& useParallelStatisticUpdate() {
            Property<IMultiThreadingConfig> property = this->getParallelStatisticUpdateConfig();
            std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
            IManualMultiThreadingConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use any multi-threading for
 * prediction.
 */
class MLRLCOMMON_API INoParallelPredictionMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoParallelPredictionMixin() override {}

        /**
         * Configures the rule learner to not use any multi-threading to predict for several query examples in parallel.
         */
        virtual void useNoParallelPrediction() {
            Property<IMultiThreadingConfig> property = this->getParallelPredictionConfig();
            property.set(std::make_unique<NoMultiThreadingConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use multi-threading to predict for
 * several examples in parallel.
 */
class MLRLCOMMON_API IParallelPredictionMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IParallelPredictionMixin() override {}

        /**
         * Configures the rule learner to use multi-threading to predict for several query examples in parallel.
         *
         * @return A reference to an object of type `IManualMultiThreadingConfig` that allows further configuration of
         *         the multi-threading behavior
         */
        virtual IManualMultiThreadingConfig& useParallelPrediction() {
            Property<IMultiThreadingConfig> property = this->getParallelPredictionConfig();
            std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
            IManualMultiThreadingConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use a stopping criterion that
 * ensures that the number of induced rules does not exceed a certain maximum.
 */
class MLRLCOMMON_API INoSizeStoppingCriterionMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoSizeStoppingCriterionMixin() override {}

        /**
         * Configures the rule learner to not use a stopping criterion that ensures that the number of induced rules
         * does not exceed a certain maximum.
         */
        virtual void useNoSizeStoppingCriterion() {
            Property<IStoppingCriterionConfig> property = this->getSizeStoppingCriterionConfig();
            property.set(std::make_unique<NoStoppingCriterionConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion that ensures
 * that the number of induced rules does not exceed a certain maximum.
 */
class MLRLCOMMON_API ISizeStoppingCriterionMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~ISizeStoppingCriterionMixin() override {}

        /**
         * Configures the rule learner to use a stopping criterion that ensures that the number of induced rules does
         * not exceed a certain maximum.
         *
         * @return A reference to an object of type `ISizeStoppingCriterionConfig` that allows further configuration of
         *         the stopping criterion
         */
        virtual ISizeStoppingCriterionConfig& useSizeStoppingCriterion() {
            Property<IStoppingCriterionConfig> property = this->getSizeStoppingCriterionConfig();
            std::unique_ptr<SizeStoppingCriterionConfig> ptr = std::make_unique<SizeStoppingCriterionConfig>();
            ISizeStoppingCriterionConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use a stopping criterion that
 * ensures that a certain time limit is not exceeded.
 */
class MLRLCOMMON_API INoTimeStoppingCriterionMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoTimeStoppingCriterionMixin() override {}

        /**
         * Configures the rule learner to not use a stopping criterion that ensures that a certain time limit is not
         * exceeded.
         */
        virtual void useNoTimeStoppingCriterion() {
            Property<IStoppingCriterionConfig> property = this->getTimeStoppingCriterionConfig();
            property.set(std::make_unique<NoStoppingCriterionConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion that ensures
 * that a certain time limit is not exceeded.
 */
class MLRLCOMMON_API ITimeStoppingCriterionMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~ITimeStoppingCriterionMixin() override {}

        /**
         * Configures the rule learner to use a stopping criterion that ensures that a certain time limit is not
         * exceeded.
         *
         * @return A reference to an object of type `ITimeStoppingCriterionConfig` that allows further configuration of
         *         the stopping criterion
         */
        virtual ITimeStoppingCriterionConfig& useTimeStoppingCriterion() {
            Property<IStoppingCriterionConfig> property = this->getTimeStoppingCriterionConfig();
            std::unique_ptr<TimeStoppingCriterionConfig> ptr = std::make_unique<TimeStoppingCriterionConfig>();
            ITimeStoppingCriterionConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion that stops
 * the induction of rules as soon as the quality of a model's predictions for the examples in the training or holdout
 * set do not improve according to a certain measure.
 */
class MLRLCOMMON_API IPrePruningMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IPrePruningMixin() override {}

        /**
         * Configures the rule learner to use a stopping criterion that stops the induction of rules as soon as the
         * quality of a model's predictions for the examples in the training or holdout set do not improve according to
         * a certain measure.
         *
         * @return A reference to an object of the type `IPrePruningConfig` that allows further configuration of the
         *         stopping criterion
         */
        virtual IPrePruningConfig& useGlobalPrePruning() {
            Property<IGlobalPruningConfig> property = this->getGlobalPruningConfig();
            std::unique_ptr<PrePruningConfig> ptr = std::make_unique<PrePruningConfig>();
            IPrePruningConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use global pruning.
 */
class MLRLCOMMON_API INoGlobalPruningMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoGlobalPruningMixin() override {}

        /**
         * Configures the rule learner to not use global pruning.
         */
        virtual void useNoGlobalPruning() {
            Property<IGlobalPruningConfig> property = this->getGlobalPruningConfig();
            property.set(std::make_unique<NoGlobalPruningConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion that keeps
 * track of the number of rules in a model that perform best with respect to the examples in the training or holdout set
 * according to a certain measure.
 */
class MLRLCOMMON_API IPostPruningMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~IPostPruningMixin() override {}

        /**
         * Configures the rule learner to use a stopping criterion that keeps track of the number of rules in a model
         * that perform best with respect to the examples in the training or holdout set according to a certain measure.
         */
        virtual IPostPruningConfig& useGlobalPostPruning() {
            Property<IGlobalPruningConfig> property = this->getGlobalPruningConfig();
            std::unique_ptr<PostPruningConfig> ptr = std::make_unique<PostPruningConfig>();
            IPostPruningConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not use a post-optimization method
 * that optimizes each rule in a model by relearning it in the context of the other rules.
 */
class MLRLCOMMON_API INoSequentialPostOptimizationMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoSequentialPostOptimizationMixin() override {}

        /**
         * Configures the rule learner to not use a post-optimization method that optimizes each rule in a model by
         * relearning it in the context of the other rules.
         */
        virtual void useNoSequentialPostOptimization() {
            Property<IPostOptimizationPhaseConfig> property = this->getSequentialPostOptimizationConfig();
            property.set(std::make_unique<NoPostOptimizationPhaseConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to use a post-optimization method that
 * optimizes each rule in a model by relearning it in the context of the other rules.
 */
class MLRLCOMMON_API ISequentialPostOptimizationMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~ISequentialPostOptimizationMixin() override {}

        /**
         * Configures the rule learner to use a post-optimization method that optimizes each rule in a model by
         * relearning it in the context of the other rules.
         *
         * @return A reference to an object of type `ISequentialPostOptimizationConfig` that allows further
         *         configuration of the post-optimization method
         */
        virtual ISequentialPostOptimizationConfig& useSequentialPostOptimization() {
            Property<IPostOptimizationPhaseConfig> property = this->getSequentialPostOptimizationConfig();
            std::unique_ptr<SequentialPostOptimizationConfig> ptr =
              std::make_unique<SequentialPostOptimizationConfig>();
            ISequentialPostOptimizationConfig& ref = *ptr;
            property.set(std::move(ptr));
            return ref;
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not calibrate marginal probabilities.
 */
class MLRLCOMMON_API INoMarginalProbabilityCalibrationMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoMarginalProbabilityCalibrationMixin() override {}

        /**
         * Configures the rule learner to not calibrate marginal probabilities.
         */
        virtual void useNoMarginalProbabilityCalibration() {
            Property<IMarginalProbabilityCalibratorConfig> property = this->getMarginalProbabilityCalibratorConfig();
            property.set(std::make_unique<NoMarginalProbabilityCalibratorConfig>());
        }
};

/**
 * Defines an interface for all classes that allow to configure a rule learner to not calibrate joint probabilities.
 */
class MLRLCOMMON_API INoJointProbabilityCalibrationMixin : virtual public IRuleLearnerConfig {
    public:

        virtual ~INoJointProbabilityCalibrationMixin() override {}

        /**
         * Configures the rule learner to not calibrate joint probabilities.
         */
        virtual void useNoJointProbabilityCalibration() {
            Property<IJointProbabilityCalibratorConfig> property = this->getJointProbabilityCalibratorConfig();
            property.set(std::make_unique<NoJointProbabilityCalibratorConfig>());
        }
};
