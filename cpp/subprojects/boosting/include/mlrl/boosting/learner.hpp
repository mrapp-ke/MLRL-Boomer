/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/boosting/binning/label_binning_auto.hpp"
#include "mlrl/boosting/binning/label_binning_equal_width.hpp"
#include "mlrl/boosting/binning/label_binning_no.hpp"
#include "mlrl/boosting/input/feature_binning_auto.hpp"
#include "mlrl/boosting/losses/loss_decomposable_logistic.hpp"
#include "mlrl/boosting/losses/loss_decomposable_squared_error.hpp"
#include "mlrl/boosting/losses/loss_decomposable_squared_hinge.hpp"
#include "mlrl/boosting/losses/loss_non_decomposable_logistic.hpp"
#include "mlrl/boosting/losses/loss_non_decomposable_squared_error.hpp"
#include "mlrl/boosting/losses/loss_non_decomposable_squared_hinge.hpp"
#include "mlrl/boosting/multi_threading/parallel_rule_refinement_auto.hpp"
#include "mlrl/boosting/multi_threading/parallel_statistic_update_auto.hpp"
#include "mlrl/boosting/post_processing/shrinkage_constant.hpp"
#include "mlrl/boosting/prediction/predictor_binary_auto.hpp"
#include "mlrl/boosting/prediction/predictor_binary_example_wise.hpp"
#include "mlrl/boosting/prediction/predictor_binary_gfm.hpp"
#include "mlrl/boosting/prediction/predictor_binary_output_wise.hpp"
#include "mlrl/boosting/prediction/predictor_probability_auto.hpp"
#include "mlrl/boosting/prediction/predictor_probability_marginalized.hpp"
#include "mlrl/boosting/prediction/predictor_probability_output_wise.hpp"
#include "mlrl/boosting/prediction/predictor_score_output_wise.hpp"
#include "mlrl/boosting/prediction/probability_calibration_isotonic.hpp"
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

namespace boosting {

    /**
     * Defines an interface for all rule learners that make use of gradient boosting.
     */
    class MLRLBOOSTING_API IBoostedRuleLearner : virtual public IRuleLearner {
        public:

            /**
             * Defines an interface for all classes that allow to configure a rule learner that makes use of gradient
             * boosting.
             */
            class IConfig : virtual public IRuleLearner::IConfig {
                    friend class AbstractBoostingRuleLearner;

                protected:

                    /**
                     * Returns an unique pointer to the configuration of the rule heads that should be induced by the
                     * rule learner.
                     *
                     * @return A reference to an unique pointer of type `IHeadConfig` that stores the configuration of
                     *         the rule heads
                     */
                    virtual std::unique_ptr<IHeadConfig>& getHeadConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the statistics that should be used by the rule
                     * learner.
                     *
                     * @return A reference to an unique pointer of type `IStatisticsConfig` that stores the
                     *         configuration of the statistics
                     */
                    virtual std::unique_ptr<IStatisticsConfig>& getStatisticsConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the L1 regularization term.
                     *
                     * @return A reference to an unique pointer of type `IRegularizationConfig` that stores the
                     *         configuration of the L1 regularization term
                     */
                    virtual std::unique_ptr<IRegularizationConfig>& getL1RegularizationConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the L2 regularization term.
                     *
                     * @return A reference to an unique pointer of type `IRegularizationConfig` that stores the
                     *         configuration of the L2 regularization term
                     */
                    virtual std::unique_ptr<IRegularizationConfig>& getL2RegularizationConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the loss function.
                     *
                     * @return A reference to an unique pointer of type `ILossConfig` that stores the configuration of
                     *         the loss function
                     */
                    virtual std::unique_ptr<ILossConfig>& getLossConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the method for the assignment of labels to
                     * bins.
                     *
                     * @return A reference to an unique pointer of type `ILabelBinningConfig` that stores the
                     *         configuration of the method for the assignment of labels to bins
                     */
                    virtual std::unique_ptr<ILabelBinningConfig>& getLabelBinningConfigPtr() = 0;

                public:

                    virtual ~IConfig() override {}
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether a holdout set should be used or not.
             */
            class IAutomaticPartitionSamplingMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticPartitionSamplingMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether a holdout set should be used or not.
                     */
                    virtual void useAutomaticPartitionSampling() {
                        std::unique_ptr<IPartitionSamplingConfig>& partitionSamplingConfigPtr =
                          this->getPartitionSamplingConfigPtr();
                        partitionSamplingConfigPtr = std::make_unique<AutomaticPartitionSamplingConfig>(
                          this->getGlobalPruningConfigPtr(), this->getMarginalProbabilityCalibratorConfigPtr(),
                          this->getJointProbabilityCalibratorConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether a method for the assignment of numerical feature values to bins should be used or not.
             */
            class IAutomaticFeatureBinningMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticFeatureBinningMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether a method for the assignment of
                     * numerical feature values to bins should be used or not.
                     */
                    virtual void useAutomaticFeatureBinning() {
                        std::unique_ptr<IFeatureBinningConfig>& featureBinningConfigPtr =
                          this->getFeatureBinningConfigPtr();
                        featureBinningConfigPtr = std::make_unique<AutomaticFeatureBinningConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether multi-threading should be used for the parallel refinement of rules or not.
             */
            class IAutomaticParallelRuleRefinementMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticParallelRuleRefinementMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether multi-threading should be used for
                     * the parallel refinement of rules or not.
                     */
                    virtual void useAutomaticParallelRuleRefinement() {
                        std::unique_ptr<IMultiThreadingConfig>& parallelRuleRefinementConfigPtr =
                          this->getParallelRuleRefinementConfigPtr();
                        parallelRuleRefinementConfigPtr = std::make_unique<AutoParallelRuleRefinementConfig>(
                          this->getLossConfigPtr(), this->getHeadConfigPtr(), this->getFeatureSamplingConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether multi-threading should be used for the parallel update of statistics or not.
             */
            class IAutomaticParallelStatisticUpdateMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticParallelStatisticUpdateMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether multi-threading should be used for
                     * the parallel update of statistics or not.
                     */
                    virtual void useAutomaticParallelStatisticUpdate() {
                        std::unique_ptr<IMultiThreadingConfig>& parallelStatisticUpdateConfigPtr =
                          this->getParallelStatisticUpdateConfigPtr();
                        parallelStatisticUpdateConfigPtr =
                          std::make_unique<AutoParallelStatisticUpdateConfig>(this->getLossConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a post processor that
             * shrinks the weights fo rules by a constant "shrinkage" parameter.
             */
            class IConstantShrinkageMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IConstantShrinkageMixin() override {}

                    /**
                     * Configures the rule learner to use a post processor that shrinks the weights of rules by a
                     * constant "shrinkage" parameter.
                     *
                     * @return A reference to an object of type `IConstantShrinkageConfig` that allows further
                     *         configuration of the loss function
                     */
                    virtual IConstantShrinkageConfig& useConstantShrinkagePostProcessor() {
                        std::unique_ptr<IPostProcessorConfig>& postProcessorConfigPtr =
                          this->getPostProcessorConfigPtr();
                        std::unique_ptr<ConstantShrinkageConfig> ptr = std::make_unique<ConstantShrinkageConfig>();
                        IConstantShrinkageConfig& ref = *ptr;
                        postProcessorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to not use L1 regularization.
             */
            class INoL1RegularizationMixin : public virtual IBoostedRuleLearner::IConfig {
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
            class IL1RegularizationMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IL1RegularizationMixin() override {}

                    /**
                     * Configures the rule learner to use L1 regularization.
                     *
                     * @return A reference to an object of type `IManualRegularizationConfig` that allows further
                     *         configuration of the regularization term
                     */
                    virtual IManualRegularizationConfig& useL1Regularization() {
                        std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr =
                          this->getL1RegularizationConfigPtr();
                        std::unique_ptr<ManualRegularizationConfig> ptr =
                          std::make_unique<ManualRegularizationConfig>();
                        IManualRegularizationConfig& ref = *ptr;
                        l1RegularizationConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to not use L2 regularization.
             */
            class INoL2RegularizationMixin : public virtual IBoostedRuleLearner::IConfig {
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
            class IL2RegularizationMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IL2RegularizationMixin() override {}

                    /**
                     * Configures the rule learner to use L2 regularization.
                     *
                     * @return A reference to an object of type `IManualRegularizationConfig` that allows further
                     *         configuration of the regularization term
                     */
                    virtual IManualRegularizationConfig& useL2Regularization() {
                        std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr =
                          this->getL2RegularizationConfigPtr();
                        std::unique_ptr<ManualRegularizationConfig> ptr =
                          std::make_unique<ManualRegularizationConfig>();
                        IManualRegularizationConfig& ref = *ptr;
                        l2RegularizationConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to not induce a default rule.
             */
            class INoDefaultRuleMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~INoDefaultRuleMixin() override {}

                    /**
                     * Configures the rule learner to not induce a default rule.
                     */
                    virtual void useNoDefaultRule() {
                        std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr = this->getDefaultRuleConfigPtr();
                        defaultRuleConfigPtr = std::make_unique<DefaultRuleConfig>(false);
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether a default rule should be induced or not.
             */
            class IAutomaticDefaultRuleMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticDefaultRuleMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether a default rule should be induced or
                     * not.
                     */
                    virtual void useAutomaticDefaultRule() {
                        std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr = this->getDefaultRuleConfigPtr();
                        defaultRuleConfigPtr = std::make_unique<AutomaticDefaultRuleConfig>(
                          this->getStatisticsConfigPtr(), this->getLossConfigPtr(), this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with complete
             * heads that predict for all available outputs.
             */
            class ICompleteHeadMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~ICompleteHeadMixin() override {}

                    /**
                     * Configures the rule learner to induce rules with complete heads that predict for all available
                     * outputs.
                     */
                    virtual void useCompleteHeads() {
                        std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                        headConfigPtr = std::make_unique<CompleteHeadConfig>(
                          this->getLabelBinningConfigPtr(), this->getParallelStatisticUpdateConfigPtr(),
                          this->getL1RegularizationConfigPtr(), this->getL2RegularizationConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with partial
             * heads that predict for a predefined number of outputs.
             */
            class IFixedPartialHeadMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IFixedPartialHeadMixin() override {}

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a predefined
                     * number of outputs.
                     *
                     * @return A reference to an object of type `IFixedPartialHeadConfig` that allows further
                     *         configuration of the rule heads
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
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with partial
             * heads that predict for a subset of the available outputs that is determined dynamically.
             */
            class IDynamicPartialHeadMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IDynamicPartialHeadMixin() override {}

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a subset of the
                     * available outputs that is determined dynamically. Only those outputs for which the square of the
                     * predictive quality exceeds a certain threshold are included in a rule head.
                     *
                     * @return A reference to an object of type `IDynamicPartialHeadConfig` that allows further
                     *         configuration of the rule heads
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
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with
             * single-output heads that predict for a single output.
             */
            class ISingleOutputHeadMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~ISingleOutputHeadMixin() override {}

                    /**
                     * Configures the rule learner to induce rules with single-output heads that predict for a single
                     * output.
                     */
                    virtual void useSingleOutputHeads() {
                        std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                        headConfigPtr = std::make_unique<SingleOutputHeadConfig>(
                          this->getLabelBinningConfigPtr(), this->getParallelStatisticUpdateConfigPtr(),
                          this->getL1RegularizationConfigPtr(), this->getL2RegularizationConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide for
             * the type of rule heads that should be used.
             */
            class IAutomaticHeadMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticHeadMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide for the type of rule heads that should be
                     * used.
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
             * Defines an interface for all classes that allow to configure a rule learner to use a dense representation
             * of gradients and Hessians.
             */
            class IDenseStatisticsMixin : public virtual IBoostedRuleLearner::IConfig {
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
             * Defines an interface for all classes that allow to configure a rule learner to use a sparse
             * representation of gradients and Hessians, if possible.
             */
            class ISparseStatisticsMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~ISparseStatisticsMixin() override {}

                    /**
                     * Configures the rule learner to use a sparse representation of gradients and Hessians, if
                     * possible.
                     */
                    virtual void useSparseStatistics() {
                        std::unique_ptr<IStatisticsConfig>& statisticsConfigPtr = this->getStatisticsConfigPtr();
                        statisticsConfigPtr = std::make_unique<SparseStatisticsConfig>(this->getLossConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether a dense or sparse representation of gradients and Hessians should be used.
             */
            class IAutomaticStatisticsMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticStatisticsMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether a dense or sparse representation of
                     * gradients and Hessians should be used.
                     */
                    virtual void useAutomaticStatistics() {
                        std::unique_ptr<IStatisticsConfig>& statisticsConfigPtr = this->getStatisticsConfigPtr();
                        statisticsConfigPtr = std::make_unique<AutomaticStatisticsConfig>(
                          this->getLossConfigPtr(), this->getHeadConfigPtr(), this->getDefaultRuleConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the logistic loss that is non-decomposable.
             */
            class INonDecomposableLogisticLossMixin : virtual public IBoostedRuleLearner::IConfig {
                public:

                    virtual ~INonDecomposableLogisticLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * logistic loss that is non-decomposable.
                     */
                    virtual void useNonDecomposableLogisticLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<NonDecomposableLogisticLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the squared error loss that is non-decomposable.
             */
            class INonDecomposableSquaredErrorLossMixin : virtual public IBoostedRuleLearner::IConfig {
                public:

                    virtual ~INonDecomposableSquaredErrorLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * squared error loss that is non-decomposable.
                     */
                    virtual void useNonDecomposableSquaredErrorLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr =
                          std::make_unique<NonDecomposableSquaredErrorLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the squared hinge loss that is non-decomposable.
             */
            class INonDecomposableSquaredHingeLossMixin : virtual public IBoostedRuleLearner::IConfig {
                public:

                    virtual ~INonDecomposableSquaredHingeLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * squared hinge loss that is non-decomposable.
                     */
                    virtual void useNonDecomposableSquaredHingeLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr =
                          std::make_unique<NonDecomposableSquaredHingeLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the logistic loss that is decomposable.
             */
            class IDecomposableLogisticLossMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IDecomposableLogisticLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * logistic loss that is applied decomposable.
                     */
                    virtual void useDecomposableLogisticLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<DecomposableLogisticLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the squared error loss that is decomposable.
             */
            class IDecomposableSquaredErrorLossMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IDecomposableSquaredErrorLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * squared error loss that is decomposable.
                     */
                    virtual void useDecomposableSquaredErrorLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<DecomposableSquaredErrorLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multivariate variant of the squared hinge loss that is decomposable.
             */
            class IDecomposableSquaredHingeLossMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IDecomposableSquaredHingeLossMixin() override {}

                    /**
                     * Configures the rule learner to use a loss function that implements a multivariate variant of the
                     * squared hinge loss that is decomposable.
                     */
                    virtual void useDecomposableSquaredHingeLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<DecomposableSquaredHingeLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to not use any method for the
             * assignment of labels to bins.
             */
            class INoLabelBinningMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~INoLabelBinningMixin() override {}

                    /**
                     * Configures the rule learner to not use any method for the assignment of labels to bins.
                     */
                    virtual void useNoLabelBinning() {
                        std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr = this->getLabelBinningConfigPtr();
                        labelBinningConfigPtr = std::make_unique<NoLabelBinningConfig>(
                          this->getL1RegularizationConfigPtr(), this->getL2RegularizationConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a method for the
             * assignment of labels to bins.
             */
            class IEqualWidthLabelBinningMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IEqualWidthLabelBinningMixin() override {}

                    /**
                     * Configures the rule learner to use a method for the assignment of labels to bins in a way such
                     * that each bin contains labels for which the predicted score is expected to belong to the same
                     * value range.
                     *
                     * @return A reference to an object of type `IEqualWidthLabelBinningConfig` that allows further
                     *         configuration of the method for the assignment of labels to bins
                     */
                    virtual IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning() {
                        std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr = this->getLabelBinningConfigPtr();
                        std::unique_ptr<EqualWidthLabelBinningConfig> ptr =
                          std::make_unique<EqualWidthLabelBinningConfig>(this->getL1RegularizationConfigPtr(),
                                                                         this->getL2RegularizationConfigPtr());
                        IEqualWidthLabelBinningConfig& ref = *ptr;
                        labelBinningConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide
             * whether a method for the assignment of labels to bins should be used or not.
             */
            class IAutomaticLabelBinningMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticLabelBinningMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide whether a method for the assignment of labels
                     * to bins should be used or not.
                     */
                    virtual void useAutomaticLabelBinning() {
                        std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr = this->getLabelBinningConfigPtr();
                        labelBinningConfigPtr = std::make_unique<AutomaticLabelBinningConfig>(
                          this->getL1RegularizationConfigPtr(), this->getL2RegularizationConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to calibrate marginal
             * probabilities via isotonic regression.
             *
             */
            class IIsotonicMarginalProbabilityCalibrationMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IIsotonicMarginalProbabilityCalibrationMixin() override {}

                    /**
                     * Configures the rule learner to calibrate marginal probabilities via isotonic regression.
                     *
                     * @return A reference to an object of type `IIsotonicMarginalProbabilityCalibratorConfig` that
                     *         allows further configuration of the calibrator
                     */
                    virtual IIsotonicMarginalProbabilityCalibratorConfig& useIsotonicMarginalProbabilityCalibration() {
                        std::unique_ptr<IMarginalProbabilityCalibratorConfig>& marginalProbabilityCalibratorConfigPtr =
                          this->getMarginalProbabilityCalibratorConfigPtr();
                        std::unique_ptr<IsotonicMarginalProbabilityCalibratorConfig> ptr =
                          std::make_unique<IsotonicMarginalProbabilityCalibratorConfig>(this->getLossConfigPtr());
                        IIsotonicMarginalProbabilityCalibratorConfig& ref = *ptr;
                        marginalProbabilityCalibratorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to calibrate joint
             * probabilities via isotonic regression.
             */
            class IIsotonicJointProbabilityCalibrationMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IIsotonicJointProbabilityCalibrationMixin() override {}

                    /**
                     * Configures the rule learner to calibrate joint probabilities via isotonic regression.
                     *
                     * @return A reference to an object of type `IIsotonicJointProbabilityCalibratorConfig` that allows
                     *         further configuration of the calibrator
                     */
                    virtual IIsotonicJointProbabilityCalibratorConfig& useIsotonicJointProbabilityCalibration() {
                        std::unique_ptr<IJointProbabilityCalibratorConfig>& jointProbabilityCalibratorConfigPtr =
                          this->getJointProbabilityCalibratorConfigPtr();
                        std::unique_ptr<IsotonicJointProbabilityCalibratorConfig> ptr =
                          std::make_unique<IsotonicJointProbabilityCalibratorConfig>(this->getLossConfigPtr());
                        IIsotonicJointProbabilityCalibratorConfig& ref = *ptr;
                        jointProbabilityCalibratorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor that
             * predicts whether individual labels of given query examples are relevant or irrelevant by discretizing the
             * individual scores or probability estimates that are predicted for each label.
             */
            class IOutputWiseBinaryPredictorMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IOutputWiseBinaryPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts whether individual labels of given
                     * query examples are relevant or irrelevant by discretizing the individual scores or probability
                     * estimates that are predicted for each label.
                     *
                     * @return A reference to an object of type `IOutputWiseBinaryPredictorConfig` that allows further
                     *         configuration of the predictor
                     */
                    virtual IOutputWiseBinaryPredictorConfig& useOutputWiseBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        std::unique_ptr<OutputWiseBinaryPredictorConfig> ptr =
                          std::make_unique<OutputWiseBinaryPredictorConfig>(this->getLossConfigPtr(),
                                                                            this->getParallelPredictionConfigPtr());
                        IOutputWiseBinaryPredictorConfig& ref = *ptr;
                        binaryPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor that
             * predicts known label vectors for given query examples by comparing the predicted scores or probability
             * estimates to the label vectors encountered in the training data.
             */
            class IExampleWiseBinaryPredictorMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IExampleWiseBinaryPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts known label vectors for given query
                     * examples by comparing the predicted scores or probability estimates to the label vectors
                     * encountered in the training data.
                     *
                     * @return A reference to an object of type `IExampleWiseBinaryPredictorConfig` that allows further
                     *         configuration of the predictor
                     */
                    virtual IExampleWiseBinaryPredictorConfig& useExampleWiseBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        std::unique_ptr<ExampleWiseBinaryPredictorConfig> ptr =
                          std::make_unique<ExampleWiseBinaryPredictorConfig>(this->getLossConfigPtr(),
                                                                             this->getParallelPredictionConfigPtr());
                        IExampleWiseBinaryPredictorConfig& ref = *ptr;
                        binaryPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor that
             * predicts whether individual labels of given query examples are relevant or irrelevant by discretizing the
             * scores or probability estimates that are predicted for each label according to the general F-measure
             * maximizer (GFM).
             */
            class IGfmBinaryPredictorMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IGfmBinaryPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts whether individual labels of given
                     * query examples are relevant or irrelevant by discretizing the scores or probability estimates
                     * that are predicted for each label according to the general F-measure maximizer (GFM).
                     *
                     * @return A reference to an object of type `IGfmBinaryPredictorConfig` that allows further
                     *         configuration of the predictor
                     */
                    virtual IGfmBinaryPredictorConfig& useGfmBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        std::unique_ptr<GfmBinaryPredictorConfig> ptr = std::make_unique<GfmBinaryPredictorConfig>(
                          this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                        IGfmBinaryPredictorConfig& ref = *ptr;
                        binaryPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide for a
             * predictor for predicting whether individual labels are relevant or irrelevant.
             */
            class IAutomaticBinaryPredictorMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticBinaryPredictorMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide for a predictor for predicting whether
                     * individual labels are relevant or irrelevant.
                     */
                    virtual void useAutomaticBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        binaryPredictorConfigPtr = std::make_unique<AutomaticBinaryPredictorConfig>(
                          this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor that
             * predicts output-wise scores for given query examples by summing up the scores that are provided by
             * individual rules for each output individually.
             */
            class IOutputWiseScorePredictorMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IOutputWiseScorePredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts output-wise scores for given query
                     * examples by summing up the scores that are provided by individual rules for each output
                     * individually.
                     */
                    virtual void useOutputWiseScorePredictor() {
                        std::unique_ptr<IScorePredictorConfig>& scorePredictorConfigPtr =
                          this->getScorePredictorConfigPtr();
                        scorePredictorConfigPtr =
                          std::make_unique<OutputWiseScorePredictorConfig>(this->getParallelPredictionConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor that
             * predicts label-wise probabilities for given query examples by transforming the individual scores that are
             * predicted for each label into probabilities.
             */
            class IOutputWiseProbabilityPredictorMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IOutputWiseProbabilityPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts label-wise probabilities for given
                     * query examples by transforming the individual scores that are predicted for each label into
                     * probabilities.
                     *
                     * @return A reference to an object of type `IOutputWiseProbabilityPredictorConfig` that allows
                     *         further configuration of the predictor
                     */
                    virtual IOutputWiseProbabilityPredictorConfig& useOutputWiseProbabilityPredictor() {
                        std::unique_ptr<IProbabilityPredictorConfig>& probabilityPredictorConfigPtr =
                          this->getProbabilityPredictorConfigPtr();
                        std::unique_ptr<OutputWiseProbabilityPredictorConfig> ptr =
                          std::make_unique<OutputWiseProbabilityPredictorConfig>(
                            this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                        IOutputWiseProbabilityPredictorConfig& ref = *ptr;
                        probabilityPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use predictor that
             * predicts label-wise probabilities for given query examples by marginalizing over the joint probabilities
             * of known label vectors.
             */
            class IMarginalizedProbabilityPredictorMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IMarginalizedProbabilityPredictorMixin() override {}

                    /**
                     * Configures the rule learner to use a predictor that predicts label-wise probabilities for given
                     * query examples by marginalizing over the joint probabilities of known label vectors.
                     *
                     * @return A reference to an object of type `IMarginalizedProbabilityPredictorConfig` that allows
                     *         further configuration of the predictor
                     */
                    virtual IMarginalizedProbabilityPredictorConfig& useMarginalizedProbabilityPredictor() {
                        std::unique_ptr<IProbabilityPredictorConfig>& probabilityPredictorConfigPtr =
                          this->getProbabilityPredictorConfigPtr();
                        std::unique_ptr<MarginalizedProbabilityPredictorConfig> ptr =
                          std::make_unique<MarginalizedProbabilityPredictorConfig>(
                            this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                        IMarginalizedProbabilityPredictorConfig& ref = *ptr;
                        probabilityPredictorConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to automatically decide for a
             * predictor for predicting probability estimates.
             */
            class IAutomaticProbabilityPredictorMixin : public virtual IBoostedRuleLearner::IConfig {
                public:

                    virtual ~IAutomaticProbabilityPredictorMixin() override {}

                    /**
                     * Configures the rule learner to automatically decide for a predictor for predicting probability
                     * estimates.
                     */
                    virtual void useAutomaticProbabilityPredictor() {
                        std::unique_ptr<IProbabilityPredictorConfig>& probabilityPredictorConfigPtr =
                          this->getProbabilityPredictorConfigPtr();
                        probabilityPredictorConfigPtr = std::make_unique<AutomaticProbabilityPredictorConfig>(
                          this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                    }
            };

            virtual ~IBoostedRuleLearner() override {}
    };

    /**
     * An abstract base class for all rule learners that makes use of gradient boosting.
     */
    class AbstractBoostingRuleLearner : public AbstractRuleLearner,
                                        virtual public IBoostedRuleLearner {
        public:

            /**
             * Allows to configure a rule learner that makes use of gradient boosting.
             */
            class Config : public AbstractRuleLearner::Config,
                           virtual public IBoostedRuleLearner::IConfig {
                protected:

                    /**
                     * An unique pointer that stores the configuration of the rule heads.
                     */
                    std::unique_ptr<IHeadConfig> headConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the statistics.
                     */
                    std::unique_ptr<IStatisticsConfig> statisticsConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the loss function.
                     */
                    std::unique_ptr<ILossConfig> lossConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the L1 regularization term.
                     */
                    std::unique_ptr<IRegularizationConfig> l1RegularizationConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the L2 regularization term.
                     */
                    std::unique_ptr<IRegularizationConfig> l2RegularizationConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the method that is used to assign labels to
                     * bins.
                     */
                    std::unique_ptr<ILabelBinningConfig> labelBinningConfigPtr_;

                private:

                    std::unique_ptr<IHeadConfig>& getHeadConfigPtr() override final;

                    std::unique_ptr<IStatisticsConfig>& getStatisticsConfigPtr() override final;

                    std::unique_ptr<IRegularizationConfig>& getL1RegularizationConfigPtr() override final;

                    std::unique_ptr<IRegularizationConfig>& getL2RegularizationConfigPtr() override final;

                    std::unique_ptr<ILossConfig>& getLossConfigPtr() override final;

                    std::unique_ptr<ILabelBinningConfig>& getLabelBinningConfigPtr() override final;

                public:

                    Config();
            };

        private:

            IBoostedRuleLearner::IConfig& config_;

            const Blas blas_;

            const Lapack lapack_;

        protected:

            /**
             * @see `AbstractRuleLearner::createStatisticsProviderFactory`
             */
            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override;

            /**
             * @see `AbstractRuleLearner::createModelBuilderFactory`
             */
            std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const override;

        public:

            /**
             * @param config        A reference to an object of type `IBoostedRuleLearner::IConfig` that specifies the
             *                      configuration that should be used by the rule learner
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
             */
            AbstractBoostingRuleLearner(IBoostedRuleLearner::IConfig& config, Blas::DdotFunction ddotFunction,
                                        Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction);
    };

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
