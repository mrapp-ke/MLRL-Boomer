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

                public:

                    virtual ~IConfig() override { };

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied example-wise.
                     *
                     * @return A reference to an object of type `ExampleWiseLogisticLossConfig` that allow further
                     *         configuration of the loss function
                     */
                    virtual ExampleWiseLogisticLossConfig& useExampleWiseLogisticLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied label-wise.
                     *
                     * @return A reference to an object of type `LabelWiseLogisticLossConfig` that allow further
                     *         configuration of the loss function
                     */
                    virtual LabelWiseLogisticLossConfig& useLabelWiseLogisticLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared error loss that is applied label-wise.
                     *
                     * @return A reference to an object of type `LabelWiseSquaredErrorLossConfig` that allow further
                     *         configuration of the loss function
                     */
                    virtual LabelWiseSquaredErrorLossConfig& useLabelWiseSquaredErrorLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared hinge loss that is applied label-wise.
                     *
                     * @return A reference to an object of type `LabelWiseSquaredHingeLossConfig` that allow further
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

                    std::unique_ptr<ILossConfig> lossConfigPtr_;

                    std::unique_ptr<ILabelBinningConfig> labelBinningConfigPtr_;

                    const ILossConfig& getLossConfig() const override;

                    const ILabelBinningConfig* getLabelBinningConfig() const override;

                public:

                    Config();

                    ExampleWiseLogisticLossConfig& useExampleWiseLogisticLoss() override;

                    LabelWiseLogisticLossConfig& useLabelWiseLogisticLoss() override;

                    LabelWiseSquaredErrorLossConfig& useLabelWiseSquaredErrorLoss() override;

                    LabelWiseSquaredHingeLossConfig& useLabelWiseSquaredHingeLoss() override;

                    void useNoLabelBinning() override;

                    EqualWidthLabelBinningConfig& useEqualWidthLabelBinning() override;

            };

        protected:

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const override;

            std::unique_ptr<IModelBuilder> createModelBuilder() const override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override;

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
