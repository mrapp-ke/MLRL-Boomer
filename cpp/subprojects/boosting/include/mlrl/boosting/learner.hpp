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
#include "mlrl/boosting/statistics/statistic_format.hpp"
#include "mlrl/boosting/statistics/statistic_format_dense.hpp"
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
             * Returns an unique pointer to the configuration of the rule heads that should be induced by the rule
             * learner.
             *
             * @return A reference to an unique pointer of type `IHeadConfig` that stores the configuration of the rule
             *         heads
             */
            virtual std::unique_ptr<IHeadConfig>& getHeadConfigPtr() = 0;

            /**
             * Returns an unique pointer to the configuration of the statistics that should be used by the rule learner.
             *
             * @return A reference to an unique pointer of type `IStatisticsConfig` that stores the configuration of the
             *         statistics
             */
            virtual std::unique_ptr<IStatisticsConfig>& getStatisticsConfigPtr() = 0;

            /**
             * Returns an unique pointer to the configuration of the L1 regularization term.
             *
             * @return A reference to an unique pointer of type `IRegularizationConfig` that stores the configuration of
             *         the L1 regularization term
             */
            virtual std::unique_ptr<IRegularizationConfig>& getL1RegularizationConfigPtr() = 0;

            /**
             * Returns an unique pointer to the configuration of the L2 regularization term.
             *
             * @return A reference to an unique pointer of type `IRegularizationConfig` that stores the configuration of
             *         the L2 regularization term
             */
            virtual std::unique_ptr<IRegularizationConfig>& getL2RegularizationConfigPtr() = 0;

            /**
             * Returns an unique pointer to the configuration of the loss function.
             *
             * @return A reference to an unique pointer of type `ILossConfig` that stores the configuration of the loss
             *         function
             */
            virtual std::unique_ptr<ILossConfig>& getLossConfigPtr() = 0;

            /**
             * Returns an unique pointer to the configuration of the method for the assignment of labels to bins.
             *
             * @return A reference to an unique pointer of type `ILabelBinningConfig` that stores the configuration of
             *         the method for the assignment of labels to bins
             */
            virtual std::unique_ptr<ILabelBinningConfig>& getLabelBinningConfigPtr() = 0;
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
                std::unique_ptr<IFeatureBinningConfig>& featureBinningConfigPtr = this->getFeatureBinningConfigPtr();
                featureBinningConfigPtr = std::make_unique<AutomaticFeatureBinningConfig>();
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
                std::unique_ptr<IMultiThreadingConfig>& parallelRuleRefinementConfigPtr =
                  this->getParallelRuleRefinementConfigPtr();
                parallelRuleRefinementConfigPtr = std::make_unique<AutoParallelRuleRefinementConfig>(
                  this->getLossConfigPtr(), this->getHeadConfigPtr(), this->getFeatureSamplingConfigPtr());
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
                std::unique_ptr<IMultiThreadingConfig>& parallelStatisticUpdateConfigPtr =
                  this->getParallelStatisticUpdateConfigPtr();
                parallelStatisticUpdateConfigPtr =
                  std::make_unique<AutoParallelStatisticUpdateConfig>(this->getLossConfigPtr());
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
                std::unique_ptr<IPostProcessorConfig>& postProcessorConfigPtr = this->getPostProcessorConfigPtr();
                std::unique_ptr<ConstantShrinkageConfig> ptr = std::make_unique<ConstantShrinkageConfig>();
                IConstantShrinkageConfig& ref = *ptr;
                postProcessorConfigPtr = std::move(ptr);
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
                std::unique_ptr<IStatisticsConfig>& statisticsConfigPtr = this->getStatisticsConfigPtr();
                statisticsConfigPtr = std::make_unique<DenseStatisticsConfig>(this->getLossConfigPtr());
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
                std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr =
                  this->getL1RegularizationConfigPtr();
                l1RegularizationConfigPtr = std::make_unique<NoRegularizationConfig>();
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
                std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr =
                  this->getL1RegularizationConfigPtr();
                std::unique_ptr<ManualRegularizationConfig> ptr = std::make_unique<ManualRegularizationConfig>();
                IManualRegularizationConfig& ref = *ptr;
                l1RegularizationConfigPtr = std::move(ptr);
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
                std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr =
                  this->getL2RegularizationConfigPtr();
                l2RegularizationConfigPtr = std::make_unique<NoRegularizationConfig>();
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
                std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr =
                  this->getL2RegularizationConfigPtr();
                std::unique_ptr<ManualRegularizationConfig> ptr = std::make_unique<ManualRegularizationConfig>();
                IManualRegularizationConfig& ref = *ptr;
                l2RegularizationConfigPtr = std::move(ptr);
                return ref;
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
                std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                headConfigPtr = std::make_unique<CompleteHeadConfig>(
                  this->getLabelBinningConfigPtr(), this->getParallelStatisticUpdateConfigPtr(),
                  this->getL1RegularizationConfigPtr(), this->getL2RegularizationConfigPtr());
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
                std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                std::unique_ptr<FixedPartialHeadConfig> ptr = std::make_unique<FixedPartialHeadConfig>(
                  this->getLabelBinningConfigPtr(), this->getParallelStatisticUpdateConfigPtr());
                IFixedPartialHeadConfig& ref = *ptr;
                headConfigPtr = std::move(ptr);
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
                std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                std::unique_ptr<DynamicPartialHeadConfig> ptr = std::make_unique<DynamicPartialHeadConfig>(
                  this->getLabelBinningConfigPtr(), this->getParallelStatisticUpdateConfigPtr());
                IDynamicPartialHeadConfig& ref = *ptr;
                headConfigPtr = std::move(ptr);
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
                std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                headConfigPtr = std::make_unique<SingleOutputHeadConfig>(
                  this->getLabelBinningConfigPtr(), this->getParallelStatisticUpdateConfigPtr(),
                  this->getL1RegularizationConfigPtr(), this->getL2RegularizationConfigPtr());
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
                std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                headConfigPtr = std::make_unique<AutomaticHeadConfig>(
                  this->getLossConfigPtr(), this->getLabelBinningConfigPtr(),
                  this->getParallelStatisticUpdateConfigPtr(), this->getL1RegularizationConfigPtr(),
                  this->getL2RegularizationConfigPtr());
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
                std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                lossConfigPtr = std::make_unique<NonDecomposableSquaredErrorLossConfig>(this->getHeadConfigPtr());
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
                std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                lossConfigPtr = std::make_unique<DecomposableSquaredErrorLossConfig>(this->getHeadConfigPtr());
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
                std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr = this->getLabelBinningConfigPtr();
                labelBinningConfigPtr = std::make_unique<NoLabelBinningConfig>(this->getL1RegularizationConfigPtr(),
                                                                               this->getL2RegularizationConfigPtr());
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
                std::unique_ptr<IScorePredictorConfig>& scorePredictorConfigPtr = this->getScorePredictorConfigPtr();
                scorePredictorConfigPtr =
                  std::make_unique<OutputWiseScorePredictorConfig>(this->getParallelPredictionConfigPtr());
            }
    };
}
