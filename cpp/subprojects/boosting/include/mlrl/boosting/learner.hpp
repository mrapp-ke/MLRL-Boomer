/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning_no.hpp"
#include "mlrl/boosting/input/feature_binning_auto.hpp"
#include "mlrl/boosting/losses/loss_decomposable_squared_error.hpp"
#include "mlrl/boosting/losses/loss_non_decomposable_squared_error.hpp"
#include "mlrl/boosting/multi_threading/parallel_rule_refinement_auto.hpp"
#include "mlrl/boosting/multi_threading/parallel_statistic_update_auto.hpp"
#include "mlrl/boosting/post_processing/shrinkage_constant.hpp"
#include "mlrl/boosting/prediction/predictor_score_output_wise.hpp"
#include "mlrl/boosting/rule_evaluation/head_type_auto.hpp"
#include "mlrl/boosting/rule_evaluation/head_type_complete.hpp"
#include "mlrl/boosting/rule_evaluation/head_type_partial_dynamic.hpp"
#include "mlrl/boosting/rule_evaluation/head_type_partial_fixed.hpp"
#include "mlrl/boosting/rule_evaluation/head_type_single.hpp"
#include "mlrl/boosting/rule_evaluation/regularization_manual.hpp"
#include "mlrl/boosting/rule_evaluation/regularization_no.hpp"
#include "mlrl/boosting/rule_model_assemblage/default_rule_auto.hpp"
#include "mlrl/boosting/sampling/partition_sampling_auto.hpp"
#include "mlrl/boosting/statistics/statistic_format.hpp"
#include "mlrl/boosting/statistics/statistic_format_auto.hpp"
#include "mlrl/boosting/statistics/statistic_format_dense.hpp"
#include "mlrl/boosting/statistics/statistic_format_sparse.hpp"
#include "mlrl/boosting/util/blas.hpp"
#include "mlrl/boosting/util/lapack.hpp"
#include "mlrl/common/learner.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a rule learner that makes use of gradient boosting.
     */
    class MLRLBOOSTING_API IBoostedRuleLearnerConfig : virtual public IRuleLearnerConfig {
        public:

            virtual ~IBoostedRuleLearnerConfig() override {}

            /**
             * Returns a `Property` that allows to access the `IHeadConfig` that stores configuration of the rule heads
             * that should be induced by the rule learner.
             *
             * @return A reference to a `Property` that allows to access the `IHeadConfig` that stores the configuration
             *         of the rule heads
             */
            virtual Property<IHeadConfig> getHeadConfig() = 0;

            /**
             * Returns a `ReadableProperty` that allows to access the `IStatisticsConfig` that stores the configuration
             * of the statistics that should be used by the rule learner.
             *
             * @return A `ReadableProperty` that allows to access the `IStatisticsConfig` that stores the configuration
             *         of the statistics
             */
            virtual ReadableProperty<IStatisticsConfig> getStatisticsConfig() const = 0;

            /**
             * Returns a `SharedProperty` that allows to access the `IClassificationStatisticsConfig` that stores the
             * configuration of the statistics that should be used by the rule learner in classification problems.
             *
             * @return A `SharedProperty` that allows to access the `IClassificationStatisticsConfig` that stores the
             *         configuration of the statistics
             */
            virtual SharedProperty<IClassificationStatisticsConfig> getClassificationStatisticsConfig() = 0;

            /**
             * Returns a `SharedProperty` that allows to access the `IRegressionStatisticsConfig` that stores the
             * configuration of the statistics that should be used by the rule learner in regression problems.
             *
             * @return A `SharedProperty` that allows to access the `IRegressionStatisticsConfig` that stores the
             *         configuration of the statistics
             */
            virtual SharedProperty<IRegressionStatisticsConfig> getRegressionStatisticsConfig() = 0;

            /**
             * Returns a `Property` that allows to access the `IRegularizationConfig` that stores the configuration of
             * the L1 regularization term.
             *
             * @return A reference to an unique pointer of type `IRegularizationConfig` that stores the configuration of
             *         the L1 regularization term
             */
            virtual Property<IRegularizationConfig> getL1RegularizationConfig() = 0;

            /**
             * Returns a `Property` that allows to access the `IRegularizationConfig` that stores the configuration of
             * the L2 regularization term.
             *
             * @return A `Property` that allows to access the `IRegularizationConfig` that stores the configuration of
             *         the L2 regularization term
             */
            virtual Property<IRegularizationConfig> getL2RegularizationConfig() = 0;

            /**
             * Returns a `ReadableProperty` that allows to access the `ILossConfig` that stores the configuration of the
             * loss function.
             *
             * @return A `ReadableProperty` that allows to access the `ILossConfig` that stores the configuration of the
             *         loss function
             */
            virtual ReadableProperty<ILossConfig> getLossConfig() const = 0;

            /**
             * Returns a `SharedProperty` that allows to access the `IClassificationLossConfig` that stores the
             * configuration of the loss function that should be used in classification problems.
             *
             * @return A `SharedProperty` that allows to access the `IClassificationLossConfig` that stores the
             *         configuration of the loss function that should be used in classification problems
             */
            virtual SharedProperty<IClassificationLossConfig> getClassificationLossConfig() = 0;

            /**
             * Returns a `SharedProperty` that allows to access the `IRegressionLossConfig` that stores the
             * configuration of the loss function that should be used in regression problems.
             *
             * @return A `SharedProperty` that allows to access the `IRegressionLossConfig` that stores the
             *         configuration of the loss function that should be used in regression problems
             */
            virtual SharedProperty<IRegressionLossConfig> getRegressionLossConfig() = 0;

            /**
             * Returns a `Property` that allows to access the `ILabelBinningConfig` that stores the configuration of the
             * method for the assignment of labels to bins.
             *
             * @return A `Property` that allows to access the `ILabelBinningConfig` that stores the configuration of the
             *         method for the assignment of labels to bins
             */
            virtual Property<ILabelBinningConfig> getLabelBinningConfig() = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide whether a
     * holdout set should be used or not.
     */
    class MLRLBOOSTING_API IAutomaticPartitionSamplingMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticPartitionSamplingMixin() override {}

            /**
             * Configures the rule learner to automatically decide whether a holdout set should be used or not.
             */
            virtual void useAutomaticPartitionSampling() {
                auto ptr = std::make_shared<AutomaticPartitionSamplingConfig>(
                  this->getGlobalPruningConfig(), this->getMarginalProbabilityCalibratorConfig(),
                  this->getJointProbabilityCalibratorConfig());
                this->getClassificationPartitionSamplingConfig().set(ptr);
                this->getRegressionPartitionSamplingConfig().set(ptr);
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide whether a
     * method for the assignment of numerical feature values to bins should be used or not.
     */
    class MLRLBOOSTING_API IAutomaticFeatureBinningMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticFeatureBinningMixin() override {}

            /**
             * Configures the rule learner to automatically decide whether a method for the assignment of numerical
             * feature values to bins should be used or not.
             */
            virtual void useAutomaticFeatureBinning() {
                this->getFeatureBinningConfig().set(std::make_unique<AutomaticFeatureBinningConfig>());
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide whether
     * multi-threading should be used for the parallel refinement of rules or not.
     */
    class MLRLBOOSTING_API IAutomaticParallelRuleRefinementMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticParallelRuleRefinementMixin() override {}

            /**
             * Configures the rule learner to automatically decide whether multi-threading should be used for the
             * parallel refinement of rules or not.
             */
            virtual void useAutomaticParallelRuleRefinement() {
                this->getParallelRuleRefinementConfig().set(std::make_unique<AutoParallelRuleRefinementConfig>(
                  this->getLossConfig(), this->getHeadConfig(), this->getFeatureSamplingConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide whether
     * multi-threading should be used for the parallel update of statistics or not.
     */
    class MLRLBOOSTING_API IAutomaticParallelStatisticUpdateMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticParallelStatisticUpdateMixin() override {}

            /**
             * Configures the rule learner to automatically decide whether multi-threading should be used for the
             * parallel update of statistics or not.
             */
            virtual void useAutomaticParallelStatisticUpdate() {
                this->getParallelStatisticUpdateConfig().set(
                  std::make_unique<AutoParallelStatisticUpdateConfig>(this->getLossConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a post processor that shrinks
     * the weights fo rules by a constant "shrinkage" parameter.
     */
    class MLRLBOOSTING_API IConstantShrinkageMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IConstantShrinkageMixin() override {}

            /**
             * Configures the rule learner to use a post processor that shrinks the weights of rules by a constant
             * "shrinkage" parameter.
             *
             * @return A reference to an object of type `IConstantShrinkageConfig` that allows further configuration of
             *         the loss function
             */
            virtual IConstantShrinkageConfig& useConstantShrinkagePostProcessor() {
                auto ptr = std::make_unique<ConstantShrinkageConfig>();
                IConstantShrinkageConfig& ref = *ptr;
                this->getPostProcessorConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a dense representation of
     * gradients and Hessians.
     */
    class MLRLBOOSTING_API IDenseStatisticsMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IDenseStatisticsMixin() override {}

            /**
             * Configures the rule learner to use a dense representation of gradients and Hessians.
             */
            virtual void useDenseStatistics() {
                auto ptr = std::make_shared<DenseStatisticsConfig>(this->getClassificationLossConfig(),
                                                                   this->getRegressionLossConfig());
                this->getClassificationStatisticsConfig().set(ptr);
                this->getRegressionStatisticsConfig().set(ptr);
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a sparse representation of
     * gradients and Hessians, if possible.
     */
    class MLRLBOOSTING_API ISparseStatisticsMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~ISparseStatisticsMixin() override {}

            /**
             * Configures the rule learner to use a sparse representation of gradients and Hessians, if possible.
             */
            virtual void useSparseStatistics() {
                auto ptr = std::make_shared<SparseStatisticsConfig>(this->getClassificationLossConfig(),
                                                                    this->getRegressionLossConfig());
                this->getClassificationStatisticsConfig().set(ptr);
                this->getRegressionStatisticsConfig().set(ptr);
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide whether a
     * dense or sparse representation of gradients and Hessians should be used.
     */
    class MLRLBOOSTING_API IAutomaticStatisticsMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticStatisticsMixin() override {}

            /**
             * Configures the rule learner to automatically decide whether a dense or sparse representation of gradients
             * and Hessians should be used.
             */
            virtual void useAutomaticStatistics() {
                auto ptr = std::make_shared<AutomaticStatisticsConfig>(
                  this->getClassificationLossConfig(), this->getRegressionLossConfig(), this->getHeadConfig(),
                  this->getDefaultRuleConfig());
                this->getClassificationStatisticsConfig().set(ptr);
                this->getRegressionStatisticsConfig().set(ptr);
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to not use L1 regularization.
     */
    class MLRLBOOSTING_API INoL1RegularizationMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~INoL1RegularizationMixin() override {}

            /**
             * Configures the rule learner to not use L1 regularization.
             */
            virtual void useNoL1Regularization() {
                this->getL1RegularizationConfig().set(std::make_unique<NoRegularizationConfig>());
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use L1 regularization.
     */
    class MLRLBOOSTING_API IL1RegularizationMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IL1RegularizationMixin() override {}

            /**
             * Configures the rule learner to use L1 regularization.
             *
             * @return A reference to an object of type `IManualRegularizationConfig` that allows further configuration
             *         of the regularization term
             */
            virtual IManualRegularizationConfig& useL1Regularization() {
                auto ptr = std::make_unique<ManualRegularizationConfig>();
                IManualRegularizationConfig& ref = *ptr;
                this->getL1RegularizationConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to not use L2 regularization.
     */
    class MLRLBOOSTING_API INoL2RegularizationMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~INoL2RegularizationMixin() override {}

            /**
             * Configures the rule learner to not use L2 regularization.
             */
            virtual void useNoL2Regularization() {
                this->getL2RegularizationConfig().set(std::make_unique<NoRegularizationConfig>());
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use L2 regularization.
     */
    class MLRLBOOSTING_API IL2RegularizationMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IL2RegularizationMixin() override {}

            /**
             * Configures the rule learner to use L2 regularization.
             *
             * @return A reference to an object of type `IManualRegularizationConfig` that allows further configuration
             *         of the regularization term
             */
            virtual IManualRegularizationConfig& useL2Regularization() {
                std::unique_ptr<ManualRegularizationConfig> ptr = std::make_unique<ManualRegularizationConfig>();
                IManualRegularizationConfig& ref = *ptr;
                this->getL2RegularizationConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to not induce a default rule.
     */
    class MLRLBOOSTING_API INoDefaultRuleMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~INoDefaultRuleMixin() override {}

            /**
             * Configures the rule learner to not induce a default rule.
             */
            virtual void useNoDefaultRule() {
                this->getDefaultRuleConfig().set(std::make_unique<DefaultRuleConfig>(false));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide whether a
     * default rule should be induced or not.
     */
    class MLRLBOOSTING_API IAutomaticDefaultRuleMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticDefaultRuleMixin() override {}

            /**
             * Configures the rule learner to automatically decide whether a default rule should be induced or not.
             */
            virtual void useAutomaticDefaultRule() {
                this->getDefaultRuleConfig().set(std::make_unique<AutomaticDefaultRuleConfig>(
                  this->getStatisticsConfig(), this->getLossConfig(), this->getHeadConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to induce rules with complete heads
     * that predict for all available outputs.
     */
    class MLRLBOOSTING_API ICompleteHeadMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~ICompleteHeadMixin() override {}

            /**
             * Configures the rule learner to induce rules with complete heads that predict for all available outputs.
             */
            virtual void useCompleteHeads() {
                this->getHeadConfig().set(std::make_unique<CompleteHeadConfig>(
                  this->getLabelBinningConfig(), this->getParallelStatisticUpdateConfig(),
                  this->getL1RegularizationConfig(), this->getL2RegularizationConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to induce rules with partial heads
     * that predict for a predefined number of outputs.
     */
    class MLRLBOOSTING_API IFixedPartialHeadMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IFixedPartialHeadMixin() override {}

            /**
             * Configures the rule learner to induce rules with partial heads that predict for a predefined number of
             * outputs.
             *
             * @return A reference to an object of type `IFixedPartialHeadConfig` that allows further configuration of
             *         the rule heads
             */
            virtual IFixedPartialHeadConfig& useFixedPartialHeads() {
                auto ptr = std::make_unique<FixedPartialHeadConfig>(this->getLabelBinningConfig(),
                                                                    this->getParallelStatisticUpdateConfig());
                IFixedPartialHeadConfig& ref = *ptr;
                this->getHeadConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to induce rules with partial heads
     * that predict for a subset of the available outputs that is determined dynamically.
     */
    class MLRLBOOSTING_API IDynamicPartialHeadMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IDynamicPartialHeadMixin() override {}

            /**
             * Configures the rule learner to induce rules with partial heads that predict for a subset of the available
             * outputs that is determined dynamically. Only those outputs for which the square of the predictive quality
             * exceeds a certain threshold are included in a rule head.
             *
             * @return A reference to an object of type `IDynamicPartialHeadConfig` that allows further configuration of
             *         the rule heads
             */
            virtual IDynamicPartialHeadConfig& useDynamicPartialHeads() {
                auto ptr = std::make_unique<DynamicPartialHeadConfig>(this->getLabelBinningConfig(),
                                                                      this->getParallelStatisticUpdateConfig());
                IDynamicPartialHeadConfig& ref = *ptr;
                this->getHeadConfig().set(std::move(ptr));
                return ref;
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to induce rules with single-output
     * heads that predict for a single output.
     */
    class MLRLBOOSTING_API ISingleOutputHeadMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~ISingleOutputHeadMixin() override {}

            /**
             * Configures the rule learner to induce rules with single-output heads that predict for a single output.
             */
            virtual void useSingleOutputHeads() {
                this->getHeadConfig().set(std::make_unique<SingleOutputHeadConfig>(
                  this->getLabelBinningConfig(), this->getParallelStatisticUpdateConfig(),
                  this->getL1RegularizationConfig(), this->getL2RegularizationConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to automatically decide for the type
     * of rule heads that should be used.
     */
    class MLRLBOOSTING_API IAutomaticHeadMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IAutomaticHeadMixin() override {}

            /**
             * Configures the rule learner to automatically decide for the type of rule heads that should be used.
             */
            virtual void useAutomaticHeads() {
                this->getHeadConfig().set(std::make_unique<AutomaticHeadConfig>(
                  this->getLossConfig(), this->getLabelBinningConfig(), this->getParallelStatisticUpdateConfig(),
                  this->getL1RegularizationConfig(), this->getL2RegularizationConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
     * implements a multivariate variant of the squared error loss that is non-decomposable.
     */
    class MLRLBOOSTING_API INonDecomposableSquaredErrorLossMixin : virtual public IBoostedRuleLearnerConfig {
        public:

            virtual ~INonDecomposableSquaredErrorLossMixin() override {}

            /**
             * Configures the rule learner to use a loss function that implements a multivariate variant of the squared
             * error loss that is non-decomposable.
             */
            virtual void useNonDecomposableSquaredErrorLoss() {
                auto ptr = std::make_shared<NonDecomposableSquaredErrorLossConfig>(this->getHeadConfig());
                this->getClassificationLossConfig().set(ptr);
                this->getRegressionLossConfig().set(ptr);
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
     * implements a multivariate variant of the squared error loss that is decomposable.
     */
    class MLRLBOOSTING_API IDecomposableSquaredErrorLossMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IDecomposableSquaredErrorLossMixin() override {}

            /**
             * Configures the rule learner to use a loss function that implements a multivariate variant of the squared
             * error loss that is decomposable.
             */
            virtual void useDecomposableSquaredErrorLoss() {
                auto ptr = std::make_shared<DecomposableSquaredErrorLossConfig>(this->getHeadConfig());
                this->getClassificationLossConfig().set(ptr);
                this->getRegressionLossConfig().set(ptr);
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to not use any method for the
     * assignment of labels to bins.
     */
    class MLRLBOOSTING_API INoLabelBinningMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~INoLabelBinningMixin() override {}

            /**
             * Configures the rule learner to not use any method for the assignment of labels to bins.
             */
            virtual void useNoLabelBinning() {
                this->getLabelBinningConfig().set(std::make_unique<NoLabelBinningConfig>(
                  this->getL1RegularizationConfig(), this->getL2RegularizationConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner to use a predictor that predicts
     * output-wise scores for given query examples by summing up the scores that are provided by individual rules for
     * each output individually.
     */
    class MLRLBOOSTING_API IOutputWiseScorePredictorMixin : public virtual IBoostedRuleLearnerConfig {
        public:

            virtual ~IOutputWiseScorePredictorMixin() override {}

            /**
             * Configures the rule learner to use a predictor that predicts output-wise scores for given query examples
             * by summing up the scores that are provided by individual rules for each output individually.
             */
            virtual void useOutputWiseScorePredictor() {
                this->getScorePredictorConfig().set(
                  std::make_unique<OutputWiseScorePredictorConfig>(this->getParallelPredictionConfig()));
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a rule learner that makes use of gradient boosting
     * to use a simple default configuration.
     */
    class MLRLBOOSTING_API IBoostedRuleLearnerMixin : virtual public IRuleLearnerMixin,
                                                      virtual public IDefaultRuleMixin,
                                                      virtual public INoL1RegularizationMixin,
                                                      virtual public INoL2RegularizationMixin,
                                                      virtual public INoLabelBinningMixin {
        public:

            virtual ~IBoostedRuleLearnerMixin() override {}

            /**
             * @see `IRuleLearnerConfig::useDefaults`
             */
            virtual void useDefaults() override {
                IRuleLearnerMixin::useDefaults();
                this->useNoL1Regularization();
                this->useNoL2Regularization();
                this->useNoLabelBinning();
            }
    };
}
