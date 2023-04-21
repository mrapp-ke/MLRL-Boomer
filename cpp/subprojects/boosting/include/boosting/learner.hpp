/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/losses/loss_example_wise_logistic.hpp"
#include "boosting/losses/loss_example_wise_squared_error.hpp"
#include "boosting/losses/loss_example_wise_squared_hinge.hpp"
#include "boosting/losses/loss_label_wise_squared_error.hpp"
#include "boosting/losses/loss_label_wise_squared_hinge.hpp"
#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"
#include "boosting/post_processing/shrinkage_constant.hpp"
#include "boosting/prediction/predictor_binary_example_wise.hpp"
#include "boosting/prediction/predictor_binary_gfm.hpp"
#include "boosting/prediction/predictor_binary_label_wise.hpp"
#include "boosting/prediction/predictor_probability_marginalized.hpp"
#include "boosting/prediction/probability_calibration_isotonic.hpp"
#include "boosting/rule_evaluation/head_type_partial_dynamic.hpp"
#include "boosting/rule_evaluation/head_type_partial_fixed.hpp"
#include "boosting/rule_evaluation/head_type_single.hpp"
#include "boosting/rule_evaluation/regularization_manual.hpp"
#include "boosting/statistics/statistic_format.hpp"
#include "boosting/statistics/statistic_format_sparse.hpp"
#include "common/learner.hpp"

namespace boosting {

    /**
     * Defines an interface for all rule learners that make use of gradient boosting.
     */
    class MLRLBOOSTING_API IBoostingRuleLearner : virtual public IRuleLearner {
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

                    virtual ~IConfig() override {};

                    /**
                     * Configures the rule learner to induce rules with complete heads that predict for all available
                     * labels.
                     */
                    virtual void useCompleteHeads() = 0;

                    /**
                     * Configures the rule learner to use a dense representation of gradients and Hessians.
                     */
                    virtual void useDenseStatistics() = 0;

                    /**
                     * Configures the rule learner to not use L1 regularization.
                     */
                    virtual void useNoL1Regularization() = 0;

                    /**
                     * Configures the rule learner to not use L2 regularization.
                     */
                    virtual void useNoL2Regularization() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied label-wise.
                     */
                    virtual void useLabelWiseLogisticLoss() = 0;

                    /**
                     * Configures the rule learner to not use any method for the assignment of labels to bins.
                     */
                    virtual void useNoLabelBinning() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting whether individual labels are
                     * relevant or irrelevant by summing up the scores that are provided by the individual rules of an
                     * existing rule-based model and transforming them into binary values according to a certain
                     * threshold that is applied to each label individually.
                     *
                     * @return A reference to an object of type `ILabelWiseBinaryPredictorConfig` that allows further
                     *         configuration of the predictor
                     */
                    virtual ILabelWiseBinaryPredictorConfig& useLabelWiseBinaryPredictor() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting regression scores by summing up the
                     * scores that are provided by the individual rules of an existing rule-based model for each label
                     * individually.
                     */
                    virtual void useLabelWiseScorePredictor() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting probability estimates by summing up
                     * the scores that are provided by individual rules of an existing rule-based model and transforming
                     * the aggregated scores into probabilities according to a certain transformation function that is
                     * applied to each label individually.
                     */
                    virtual void useLabelWiseProbabilityPredictor() = 0;
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a post processor that
             * shrinks the weights fo rules by a "shrinkage" parameter.
             */
            class IShrinkageMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IShrinkageMixin() override {};

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
             * Defines an interface for all classes that allow to configure a rule learner to use regularization.
             */
            class IRegularizationMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IRegularizationMixin() override {};

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
            class INoDefaultRuleMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~INoDefaultRuleMixin() override {};

                    /**
                     * Configures the rule learner to not induce a default rule.
                     */
                    virtual void useNoDefaultRule() {
                        std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr = this->getDefaultRuleConfigPtr();
                        defaultRuleConfigPtr = std::make_unique<DefaultRuleConfig>(false);
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with partial
             * heads.
             */
            class IPartialHeadMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IPartialHeadMixin() override {};

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a predefined
                     * number of labels.
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

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a subset of the
                     * available labels that is determined dynamically. Only those labels for which the square of the
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

                    /**
                     * Configures the rule learner to induce rules with single-label heads that predict for a single
                     * label.
                     */
                    virtual void useSingleLabelHeads() {
                        std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                        headConfigPtr = std::make_unique<SingleLabelHeadConfig>(
                          this->getLabelBinningConfigPtr(), this->getParallelStatisticUpdateConfigPtr(),
                          this->getL1RegularizationConfigPtr(), this->getL2RegularizationConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a sparse
             * representation of gradients and Hessians, if possible.
             */
            class ISparseStatisticsMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~ISparseStatisticsMixin() override {};

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
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multi-label variant of the logistic loss that is applied example-wise.
             */
            class IExampleWiseLogisticLossMixin : virtual public IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IExampleWiseLogisticLossMixin() override {};

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied example-wise.
                     */
                    virtual void useExampleWiseLogisticLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<ExampleWiseLogisticLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multi-label variant of the squared error loss that is applied example-wise.
             */
            class IExampleWiseSquaredErrorLossMixin : virtual public IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IExampleWiseSquaredErrorLossMixin() override {};

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared error loss that is applied example-wise.
                     */
                    virtual void useExampleWiseSquaredErrorLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<ExampleWiseSquaredErrorLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multi-label variant of the squared hinge loss that is applied example-wise.
             */
            class IExampleWiseSquaredHingeLossMixin : virtual public IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IExampleWiseSquaredHingeLossMixin() override {};

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared hinge loss that is applied example-wise.
                     */
                    virtual void useExampleWiseSquaredHingeLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<ExampleWiseSquaredHingeLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multi-label variant of the squared error loss that is applied label-wise.
             */
            class ILabelWiseSquaredErrorLossMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~ILabelWiseSquaredErrorLossMixin() override {};

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared error loss that is applied label-wise.
                     */
                    virtual void useLabelWiseSquaredErrorLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<LabelWiseSquaredErrorLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multi-label variant of the squared hinge loss that is applied label-wise.
             */
            class ILabelWiseSquaredHingeLossMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~ILabelWiseSquaredHingeLossMixin() override {};

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared hinge loss that is applied label-wise.
                     */
                    virtual void useLabelWiseSquaredHingeLoss() {
                        std::unique_ptr<ILossConfig>& lossConfigPtr = this->getLossConfigPtr();
                        lossConfigPtr = std::make_unique<LabelWiseSquaredHingeLossConfig>(this->getHeadConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a method for the
             * assignment of labels to bins.
             */
            class ILabelBinningMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~ILabelBinningMixin() override {};

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
             * Defines an interface for all classes that allow to configure a rule learner to calibrate marginal
             * probabilities via isotonic regression.
             *
             */
            class IIsotonicMarginalProbabilityCalibrationMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IIsotonicMarginalProbabilityCalibrationMixin() override {};

                    /**
                     * Configures the rule learner to calibrate marginal probabilities via isotonic regression.
                     */
                    virtual void useIsotonicMarginalProbabilityCalibration() {
                        std::unique_ptr<IMarginalProbabilityCalibratorConfig>& marginalProbabilityCalibratorConfigPtr =
                          this->getMarginalProbabilityCalibratorConfigPtr();
                        marginalProbabilityCalibratorConfigPtr =
                          std::make_unique<IsotonicMarginalProbabilityCalibratorConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to calibrate joint
             * probabilities via isotonic regression.
             */
            class IIsotonicJointProbabilityCalibrationMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IIsotonicJointProbabilityCalibrationMixin() override {};

                    /**
                     * Configures the rule learner to calibrate joint probabilities via isotonic regression.
                     */
                    virtual void useIsotonicJointProbabilityCalibration() {
                        std::unique_ptr<IJointProbabilityCalibratorConfig>& jointProbabilityCalibratorConfigPtr =
                          this->getJointProbabilityCalibratorConfigPtr();
                        jointProbabilityCalibratorConfigPtr =
                          std::make_unique<IsotonicJointProbabilityCalibratorConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor for
             * predicting whether individual labels are relevant or irrelevant by summing up the scores that are
             * provided by an existing rule-based model and comparing the aggregated score vector to the known label
             * vectors according to a certain distance measure.
             */
            class IExampleWiseBinaryPredictorMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IExampleWiseBinaryPredictorMixin() override {};

                    /**
                     * Configures the rule learner to use a predictor for predicting whether individual labels are
                     * relevant or irrelevant by summing up the scores that are provided by an existing rule-based model
                     * and comparing the aggregated score vector to the known label vectors according to a certain
                     * distance measure. The label vector that is closest to the aggregated score vector is finally
                     * predicted.
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
             * Defines an interface for all classes that allow to configure a rule learner to use a predictor for
             * predicting whether individual labels are relevant or irrelevant by summing up the scores that are
             * provided by the individual rules of an existing rule-based model and transforming them into binary values
             * according to the general F-measure maximizer (GFM).
             */
            class IGfmBinaryPredictorMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IGfmBinaryPredictorMixin() override {};

                    /**
                     * Configures the rule learner to use a predictor for predicting whether individual labels are
                     * relevant or irrelevant by summing up the scores that are provided by the individual rules of a
                     * existing rule-based model and transforming them into binary values according to the general
                     * F-measure maximizer (GFM).
                     */
                    virtual void useGfmBinaryPredictor() {
                        std::unique_ptr<IBinaryPredictorConfig>& binaryPredictorConfigPtr =
                          this->getBinaryPredictorConfigPtr();
                        binaryPredictorConfigPtr = std::make_unique<GfmBinaryPredictorConfig>(
                          this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use predictor for
             * predicting probability estimates by summing up the scores that are provided by individual rules of an
             * existing rule-based model and comparing the aggregated score vector to the known label vectors according
             * to a certain distance measure.
             */
            class IMarginalizedProbabilityPredictorMixin : public virtual IBoostingRuleLearner::IConfig {
                public:

                    virtual ~IMarginalizedProbabilityPredictorMixin() override {};

                    /**
                     * Configures the rule learner to use a predictor for predicting probability estimates by summing up
                     * the scores that are provided by individual rules of an existing rule-based model and comparing
                     * the aggregated score vector to the known label vectors according to a certain distance measure.
                     * The probability for an individual label calculates as the sum of the distances that have been
                     * obtained for all label vectors, where the respective label is specified to be relevant, divided
                     * by the total sum of all distances.
                     */
                    virtual void useMarginalizedProbabilityPredictor() {
                        std::unique_ptr<IProbabilityPredictorConfig>& probabilityPredictorConfigPtr =
                          this->getProbabilityPredictorConfigPtr();
                        probabilityPredictorConfigPtr = std::make_unique<MarginalizedProbabilityPredictorConfig>(
                          this->getLossConfigPtr(), this->getParallelPredictionConfigPtr());
                    }
            };

            virtual ~IBoostingRuleLearner() override {};
    };

    /**
     * An abstract base class for all rule learners that makes use of gradient boosting.
     */
    class AbstractBoostingRuleLearner : public AbstractRuleLearner,
                                        virtual public IBoostingRuleLearner {
        public:

            /**
             * Allows to configure a rule learner that makes use of gradient boosting.
             */
            class Config : public AbstractRuleLearner::Config,
                           virtual public IBoostingRuleLearner::IConfig {
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

                    void useCompleteHeads() override;

                    void useDenseStatistics() override;

                    void useNoL1Regularization() override;

                    void useNoL2Regularization() override;

                    void useLabelWiseLogisticLoss() override;

                    void useNoLabelBinning() override;

                    ILabelWiseBinaryPredictorConfig& useLabelWiseBinaryPredictor() override;

                    void useLabelWiseScorePredictor() override;

                    void useLabelWiseProbabilityPredictor() override;
            };

        private:

            IBoostingRuleLearner::IConfig& config_;

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
             * @param config        A reference to an object of type `IBoostingRuleLearner::IConfig` that specifies the
             *                      configuration that should be used by the rule learner
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
             */
            AbstractBoostingRuleLearner(IBoostingRuleLearner::IConfig& config, Blas::DdotFunction ddotFunction,
                                        Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction);
    };

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
