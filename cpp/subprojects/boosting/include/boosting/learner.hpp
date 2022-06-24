/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning( push )
    #pragma warning( disable : 4250 )
#endif

#include "common/learner.hpp"
#include "boosting/binning/label_binning.hpp"
#include "boosting/losses/loss.hpp"
#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"
#include "boosting/post_processing/shrinkage_constant.hpp"
#include "boosting/rule_evaluation/head_type_partial_dynamic.hpp"
#include "boosting/rule_evaluation/head_type_partial_fixed.hpp"
#include "boosting/rule_evaluation/regularization_manual.hpp"
#include "boosting/statistics/statistic_format.hpp"


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

                private:

                    /**
                     * Returns the configuration of the rule heads that should be induced by the rule learner.
                     *
                     * @return A reference to an object of type `IHeadConfig` that specifies the configuration of the
                     *         rule heads
                     */
                    virtual const IHeadConfig& getHeadConfig() const = 0;

                    /**
                     * Returns the configuration of the statistics that should be used by the rule learner.
                     *
                     * @return A reference to an object of type `IStatisticsConfig` that specifies the configuration of
                     *         the statistics
                     */
                    virtual const IStatisticsConfig& getStatisticsConfig() const = 0;

                    /**
                     * Returns the configuration of the L1 regularization term.
                     *
                     * @return A reference to an object of type `IRegularizationConfig` that specifies the configuration
                     *         of the L1 regularization term
                     */
                    virtual const IRegularizationConfig& getL1RegularizationConfig() const = 0;

                    /**
                     * Returns the configuration of the L2 regularization term.
                     *
                     * @return A reference to an object of type `IRegularizationConfig` that specifies the configuration
                     *         of the L2 regularization term
                     */
                    virtual const IRegularizationConfig& getL2RegularizationConfig() const = 0;

                    /**
                     * Returns the configuration of the loss function.
                     *
                     * @return A reference to an object of type `ILossConfig` that specifies the configuration of the
                     *         loss function
                     */
                    virtual const ILossConfig& getLossConfig() const = 0;

                    /**
                     * Returns the configuration of the method for the assignment of labels to bins.
                     *
                     * @return A reference to an object of type `ILabelBinningConfig` that specifies the configuration
                     *         of the method for the assignment of labels to bins
                     */
                    virtual const ILabelBinningConfig& getLabelBinningConfig() const = 0;

                    /**
                     * Returns the configuration of the predictor that predicts whether individual labels of given query
                     * examples are relevant or irrelevant.
                     *
                     * @return A reference to an object of type `IClassificationPredictorConfig` that specifies the
                     *         configuration of the predictor that predicts whether individual labels of given query
                     *         examples are relevant or irrelevant
                     */
                    virtual const IClassificationPredictorConfig& getClassificationPredictorConfig() const = 0;

                    /**
                     * Returns the configuration of the predictor that predicts regression scores for individual labels.
                     *
                     * @return A reference to an object of type `IClassificationPredictorConfig` that specifies the
                     *         configuration of the predictor that predicts regression scores for individual labels
                     */
                    virtual const IRegressionPredictorConfig& getRegressionPredictorConfig() const = 0;

                    /**
                     * Returns the configuration of the predictor that predicts probability estimates for individual
                     * labels.
                     *
                     * @return A reference to an object of type `IClassificationPredictorConfig` that specifies the
                     *         configuration of the predictor that predicts probability estimates for individual labels
                     */
                    virtual const IProbabilityPredictorConfig& getProbabilityPredictorConfig() const = 0;

                public:

                    virtual ~IConfig() override { };

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
                     */
                    virtual void useLabelWiseClassificationPredictor() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting regression scores by summing up the
                     * scores that are provided by the individual rules of an existing rule-based model for each label
                     * individually.
                     */
                    virtual void useLabelWiseRegressionPredictor() = 0;

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
            class IShrinkageMixin {

                public:

                    virtual ~IShrinkageMixin() { };

                    /**
                     * Configures the rule learner to use a post processor that shrinks the weights of rules by a
                     * constant "shrinkage" parameter.
                     *
                     * @return A reference to an object of type `IConstantShrinkageConfig` that allows further
                     *         configuration of the loss function
                     */
                    virtual IConstantShrinkageConfig& useConstantShrinkagePostProcessor() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use regularization.
             */
            class IRegularizationMixin {

                public:

                    virtual ~IRegularizationMixin() { };

                    /**
                     * Configures the rule learner to use L1 regularization.
                     *
                     * @return A reference to an object of type `IManualRegularizationConfig` that allows further
                     *         configuration of the regularization term
                     */
                    virtual IManualRegularizationConfig& useL1Regularization() = 0;

                    /**
                     * Configures the rule learner to use L2 regularization.
                     *
                     * @return A reference to an object of type `IManualRegularizationConfig` that allows further
                     *         configuration of the regularization term
                     */
                    virtual IManualRegularizationConfig& useL2Regularization() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to not induce a default rule.
             */
            class INoDefaultRuleMixin {

                public:

                    virtual ~INoDefaultRuleMixin() { };

                    /**
                     * Configures the rule learner to not induce a default rule.
                     */
                    virtual void useNoDefaultRule() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with partial
             * heads.
             */
            class IPartialHeadMixin {

                public:

                    virtual ~IPartialHeadMixin() { };

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a predefined
                     * number of labels.
                     *
                     * @return A reference to an object of type `IFixedPartialHeadConfig` that allows further
                     *         configuration of the rule heads
                     */
                    virtual IFixedPartialHeadConfig& useFixedPartialHeads() = 0;

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a subset of the
                     * available labels that is determined dynamically. Only those labels for which the square of the
                     * predictive quality exceeds a certain threshold are included in a rule head.
                     *
                     * @return A reference to an object of type `IDynamicPartialHeadConfig` that allows further
                     *         configuration of the rule heads
                     */
                    virtual IDynamicPartialHeadConfig& useDynamicPartialHeads() = 0;

                    /**
                     * Configures the rule learner to induce rules with single-label heads that predict for a single
                     * label.
                     */
                    virtual void useSingleLabelHeads() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a sparse
             * representation of gradients and Hessians, if possible.
             */
            class ISparseStatisticsMixin {

                public:

                    virtual ~ISparseStatisticsMixin() { };

                    /**
                     * Configures the rule learner to use a sparse representation of gradients and Hessians, if
                     * possible.
                     */
                    virtual void useSparseStatistics() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multi-label variant of the logistic loss that is applied example-wise.
             */
            class IExampleWiseLogisticLossMixin {

                public:

                    virtual ~IExampleWiseLogisticLossMixin() { };

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied example-wise.
                     */
                    virtual void useExampleWiseLogisticLoss() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multi-label variant of the squared error loss that is applied label-wise.
             */
            class ILabelWiseSquaredErrorLossMixin {

                public:

                    virtual ~ILabelWiseSquaredErrorLossMixin() { };

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared error loss that is applied label-wise.
                     */
                    virtual void useLabelWiseSquaredErrorLoss() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a loss function that
             * implements a multi-label variant of the squared hinge loss that is applied label-wise.
             */
            class ILabelWiseSquaredHingeLossMixin {

                public:

                    virtual ~ILabelWiseSquaredHingeLossMixin() { };

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared hinge loss that is applied label-wise.
                     */
                    virtual void useLabelWiseSquaredHingeLoss() = 0;

            };

            virtual ~IBoostingRuleLearner() override { };

    };

    /**
     * An abstract base class for all rule learners that makes use of gradient boosting.
     */
    class AbstractBoostingRuleLearner : public AbstractRuleLearner, virtual public IBoostingRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of gradient boosting.
             */
            class Config : public AbstractRuleLearner::Config, virtual public IBoostingRuleLearner::IConfig {

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

                    /**
                     * An unique pointer that stores the configuration of the predictor that is used to predict binary
                     * labels.
                     */
                    std::unique_ptr<IClassificationPredictorConfig> classificationPredictorConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the predictor that is used to predict
                     * regression scores.
                     */
                    std::unique_ptr<IRegressionPredictorConfig> regressionPredictorConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the predictor that is used to predict
                     * probability estimates.
                     */
                    std::unique_ptr<IProbabilityPredictorConfig> probabilityPredictorConfigPtr_;

                private:

                    const IHeadConfig& getHeadConfig() const override final;

                    const IStatisticsConfig& getStatisticsConfig() const override final;

                    const IRegularizationConfig& getL1RegularizationConfig() const override final;

                    const IRegularizationConfig& getL2RegularizationConfig() const override final;

                    const ILossConfig& getLossConfig() const override final;

                    const ILabelBinningConfig& getLabelBinningConfig() const override final;

                    const IClassificationPredictorConfig& getClassificationPredictorConfig() const override final;

                    const IRegressionPredictorConfig& getRegressionPredictorConfig() const override final;

                    const IProbabilityPredictorConfig& getProbabilityPredictorConfig() const override final;

                public:

                    Config();

                    void useCompleteHeads() override;

                    void useDenseStatistics() override;

                    void useNoL1Regularization() override;

                    void useNoL2Regularization() override;

                    void useLabelWiseLogisticLoss() override;

                    void useNoLabelBinning() override;

                    void useLabelWiseClassificationPredictor() override;

                    void useLabelWiseRegressionPredictor() override;

                    void useLabelWiseProbabilityPredictor() override;

            };

        private:

            const IBoostingRuleLearner::IConfig& config_;

            Blas blas_;

            Lapack lapack_;

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

            /**
             * @see `AbstractRuleLearner::createLabelSpaceInfo`
             */
            std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo(
                const IRowWiseLabelMatrix& labelMatrix) const override;

            /**
             * @see `AbstractRuleLearner::createClassificationPredictorFactory`
             */
            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `AbstractRuleLearner::createRegressionPredictorFactory`
             */
            std::unique_ptr<IRegressionPredictorFactory> createRegressionPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `AbstractRuleLearner::createProbabilityPredictorFactory`
             */
            std::unique_ptr<IProbabilityPredictorFactory> createProbabilityPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

        public:

            /**
             * @param config        A reference to an object of type `IBoostingRuleLearner::IConfig` that specifies the
             *                      configuration that should be used by the rule learner
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
             */
            AbstractBoostingRuleLearner(const IBoostingRuleLearner::IConfig& config, Blas::DdotFunction ddotFunction,
                                        Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction);

    };

}

#ifdef _WIN32
    #pragma warning ( pop )
#endif
