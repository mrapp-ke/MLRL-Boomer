/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/learner.hpp"
#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/losses/loss_example_wise_logistic.hpp"
#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/losses/loss_label_wise_squared_error.hpp"
#include "boosting/losses/loss_label_wise_squared_hinge.hpp"
#include "boosting/output/predictor_classification_example_wise.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/output/predictor_regression_label_wise.hpp"
#include "boosting/output/predictor_probability_label_wise.hpp"
#include "boosting/post_processing/shrinkage_constant.hpp"


namespace boosting {

    /**
     * Defines an interface for all rule learners that make use of gradient boosting.
     */
    class IBoostingRuleLearner : virtual public IRuleLearner {

        public:

            /**
             * Defines an interface for all classes that allow to configure a rule learner that makes use of gradient
             * boosting.
             */
            class IConfig : virtual public IRuleLearner::IConfig {

                friend class BoostingRuleLearner;

                private:

                    /**
                     * Returns the configuration of the method that post-processes the predictions of rules once they
                     * have been learned.
                     *
                     * @return A reference to an object of type `IPostProcessorConfig` that specifies the configuration
                     *         of the method that post-processes the predictions of rules once they have been learned
                     */
                    virtual const IPostProcessorConfig& getPostProcessorConfig() const = 0;

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
                     * @return A pointer to an object of type `ILabelBinningConfig` that specifies the configuration of
                     *         the algorithm for the assignment of labels to bins or a null pointer, if no such method
                     *         should be used
                     */
                    virtual const ILabelBinningConfig* getLabelBinningConfig() const = 0;

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
                     * Configures the rule learner to not use any post processor.
                     */
                    virtual void useNoPostProcessor() = 0;

                    /**
                     * Configures the rule learner to use a post processor that shrinks the weights of rules by a
                     * constant "shrinkage" parameter.
                     *
                     * @return A reference to an object of type `ConstantShrinkageConfig` that allows further
                     *         configuration of the loss function
                     */
                    virtual ConstantShrinkageConfig& useConstantShrinkagePostProcessor() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied example-wise.
                     *
                     * @return A reference to an object of type `ExampleWiseLogisticLossConfig` that allows further
                     *         configuration of the loss function
                     */
                    virtual ExampleWiseLogisticLossConfig& useExampleWiseLogisticLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied label-wise.
                     *
                     * @return A reference to an object of type `LabelWiseLogisticLossConfig` that allows further
                     *         configuration of the loss function
                     */
                    virtual LabelWiseLogisticLossConfig& useLabelWiseLogisticLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared error loss that is applied label-wise.
                     *
                     * @return A reference to an object of type `LabelWiseSquaredErrorLossConfig` that allows further
                     *         configuration of the loss function
                     */
                    virtual LabelWiseSquaredErrorLossConfig& useLabelWiseSquaredErrorLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared hinge loss that is applied label-wise.
                     *
                     * @return A reference to an object of type `LabelWiseSquaredHingeLossConfig` that allows further
                     *         configuration of the loss function
                     */
                    virtual LabelWiseSquaredHingeLossConfig& useLabelWiseSquaredHingeLoss() = 0;

                    /**
                     * Configures the algorithm to not use any method for the assignment of labels to bins.
                     */
                    virtual void useNoLabelBinning() = 0;

                    /**
                     * Configures the algorithm to use a method for the assignment of labels to bins in a way such that
                     * each bin contains labels for which the predicted score is expected to belong to the same value
                     * range.
                     *
                     * @return A reference to an object of type `EqualWidthLabelBinningConfig` that allows further
                     *         configuration of the method for the assignment of labels to bins
                     */
                    virtual EqualWidthLabelBinningConfig& useEqualWidthLabelBinning() = 0;

                    /**
                     * Configures the algorithm to use a predictor for predicting whether individual labels are relevant
                     * or irrelevant by summing up the scores that are provided by an existing rule-based model and
                     * comparing the aggregated score vector to the known label vectors according to a certain distance
                     * measure. The label vector that is closest to the aggregated score vector is finally predicted.
                     *
                     * @return A reference to an object of type `ExampleWiseClassificationPredictorConfig` that allows
                     *         further configuration of the predictor for predicting whether individual labels are
                     *         relevant or irrelevant
                     */
                    virtual ExampleWiseClassificationPredictorConfig& useExampleWiseClassificationPredictor() = 0;

                    /**
                     * Configures the algorithm to use a predictor for predicting whether individual labels are relevant
                     * or irrelevant by summing up the scores that are provided by the individual rules of an existing
                     * rule-based model and transforming them into binary values according to a certain threshold that
                     * is applied to each label individually.
                     *
                     * @return A reference to an object of type `LabelWiseClassificationPredictorConfig` that allows
                     *         further configuration of the predictor for predicting whether individual labels are
                     *         relevant or irrelevant
                     */
                    virtual LabelWiseClassificationPredictorConfig& useLabelWiseClassificationPredictor() = 0;

                    /**
                     * Configures the algorithm to use a predictor for predicting regression scores by summing up the
                     * scores that are provided by the individual rules of an existing rule-based model for each label
                     * individually.
                     *
                     * @return A reference to an object of type `LabelWiseRegressionPredictorConfig` that allows further
                     *         configuration of the predictor for predicting regression scores
                     */
                    virtual LabelWiseRegressionPredictorConfig& useLabelWiseRegressionPredictor() = 0;

                    /**
                     * Configures the algorithm to use a predictor for predicting probability estimates by summing up
                     * the scores that are provided by individual rules of an existing rule-based models and
                     * transforming the aggregated scores into probabilities according to a certain transformation
                     * function that is applied to each label individually.
                     *
                     * @return A reference to an object of type `LabelWiseProbabilityPredictorConfig` that allows
                     *         further configuration of the predictor for predicting probability estimates
                     */
                    virtual LabelWiseProbabilityPredictorConfig& useLabelWiseProbabilityPredictor() = 0;

            };

            virtual ~IBoostingRuleLearner() override { };

    };

    /**
     * A rule learner that makes use of gradient boosting.
     */
    class BoostingRuleLearner final : public AbstractRuleLearner, virtual public IBoostingRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of gradient boosting.
             */
            class Config : public AbstractRuleLearner::Config, virtual public IBoostingRuleLearner::IConfig {

                private:

                    std::unique_ptr<IPostProcessorConfig> postProcessorConfigPtr_;

                    std::unique_ptr<ILossConfig> lossConfigPtr_;

                    std::unique_ptr<ILabelBinningConfig> labelBinningConfigPtr_;

                    std::unique_ptr<IClassificationPredictorConfig> classificationPredictorConfigPtr_;

                    std::unique_ptr<IRegressionPredictorConfig> regressionPredictorConfigPtr_;

                    std::unique_ptr<IProbabilityPredictorConfig> probabilityPredictorConfigPtr_;

                    const IPostProcessorConfig& getPostProcessorConfig() const override;

                    const ILossConfig& getLossConfig() const override;

                    const ILabelBinningConfig* getLabelBinningConfig() const override;

                    const IClassificationPredictorConfig& getClassificationPredictorConfig() const override;

                    const IRegressionPredictorConfig& getRegressionPredictorConfig() const override;

                    const IProbabilityPredictorConfig& getProbabilityPredictorConfig() const override;

                public:

                    Config();

                    void useNoPostProcessor() override;

                    ConstantShrinkageConfig& useConstantShrinkagePostProcessor() override;

                    ExampleWiseLogisticLossConfig& useExampleWiseLogisticLoss() override;

                    LabelWiseLogisticLossConfig& useLabelWiseLogisticLoss() override;

                    LabelWiseSquaredErrorLossConfig& useLabelWiseSquaredErrorLoss() override;

                    LabelWiseSquaredHingeLossConfig& useLabelWiseSquaredHingeLoss() override;

                    void useNoLabelBinning() override;

                    EqualWidthLabelBinningConfig& useEqualWidthLabelBinning() override;

                    ExampleWiseClassificationPredictorConfig& useExampleWiseClassificationPredictor() override;

                    LabelWiseClassificationPredictorConfig& useLabelWiseClassificationPredictor() override;

                    LabelWiseRegressionPredictorConfig& useLabelWiseRegressionPredictor() override;

                    LabelWiseProbabilityPredictorConfig& useLabelWiseProbabilityPredictor() override;

            };

        private:

            std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr_;

        protected:

            std::unique_ptr<IPostProcessorFactory> createPostProcessorFactory() const override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const override;

            std::unique_ptr<IModelBuilder> createModelBuilder() const override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override;

            std::unique_ptr<IRegressionPredictorFactory> createRegressionPredictorFactory() const override;

            std::unique_ptr<IProbabilityPredictorFactory> createProbabilityPredictorFactory() const override;

        public:

            /**
             * @param configPtr An unique pointer to an object of type `IBoostingRuleLearner::IConfig` that specifies
             *                  the configuration that should be used by the rule learner
             */
            BoostingRuleLearner(std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr);

    };

    /**
     * Creates and returns a new object of type `IBoostingRuleLearner::IConfig`.
     *
     * @return An unique pointer to an object of type `IBoostingRuleLearner::IConfig` that has been created
     */
    std::unique_ptr<IBoostingRuleLearner::IConfig> createBoostingRuleLearnerConfig();

    /**
     * Creates and returns a new object of type `IBoostingRuleLearner`.
     *
     * @param configPtr An unique pointer to an object of type `IBoostingRuleLearner::IConfig` that specifies the
     *                  configuration that should be used by the rule learner.
     * @return          An unique pointer to an object of type `IBoostingRuleLearner` that has been created
     */
    std::unique_ptr<IBoostingRuleLearner> createBoostingRuleLearner(
        std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr);

}
