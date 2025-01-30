/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/learner.hpp"
#include "mlrl/boosting/model/rule_list_builder.hpp"
#include "mlrl/boosting/rule_evaluation/rule_compare_function.hpp"
#include "mlrl/common/learner_common.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * Allows to configure the individual modules of a rule learner that makes use of gradient boosting, depending on an
     * `IBoostedRuleLearnerConfig`.
     */
    class BoostedRuleLearnerConfigurator final : public RuleLearnerConfigurator {
        private:

            const std::unique_ptr<IBoostedRuleLearnerConfig> configPtr_;

            const BlasFactory blasFactory_;

            const LapackFactory lapackFactory_;

        public:

            /**
             * @param configPtr             An unique pointer to an object of type `IBoostedRuleLearnerConfig`
             * @param float32BlasRoutines   A reference to an object of type `Blas::Routines` that stores function
             *                              pointers to all supported BLAS routines operating on 32-bit floating point
             *                              values
             * @param float64BlasRoutines   A reference to an object of type `Blas::Routines` that stores function
             *                              pointers to all supported BLAS routines operating on 64-bit floating point
             *                              values
             * @param float32LapackRoutines A reference to an object of type `Lapack::Routines` that stores function
             *                              pointers to all supported LAPACK routines operating on 32-bit floating point
             *                              values
             * @param float64LapackRoutines A reference to an object of type `Lapack::Routines` that stores function
             *                              pointers to all supported LAPACK routines operating on 64-bit floating point
             *                              values
             */
            BoostedRuleLearnerConfigurator(std::unique_ptr<IBoostedRuleLearnerConfig> configPtr,
                                           const Blas<float32>::Routines& float32BlasRoutines,
                                           const Blas<float64>::Routines& float64BlasRoutines,
                                           const Lapack<float32>::Routines& float32LapackRoutines,
                                           const Lapack<float64>::Routines& float64LapackRoutines)
                : RuleLearnerConfigurator(*configPtr), configPtr_(std::move(configPtr)),
                  blasFactory_(float32BlasRoutines, float64BlasRoutines),
                  lapackFactory_(float32LapackRoutines, float64LapackRoutines) {}

            /**
             * @see `RuleLearnerConfigurator::createModelBuilderFactory`
             */
            std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const override {
                return std::make_unique<RuleListBuilderFactory>();
            }

            /**
             * @see `RuleLearnerConfigurator::createClassificationStatisticsProviderFactory`
             */
            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override {
                return configPtr_->getStatisticTypeConfig().get().createClassificationStatisticsProviderFactory(
                  featureMatrix, labelMatrix, blasFactory_, lapackFactory_);
            }

            /**
             * @see `RuleLearnerConfigurator::createRegressionStatisticsProviderFactory`
             */
            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix) const override {
                return configPtr_->getStatisticTypeConfig().get().createRegressionStatisticsProviderFactory(
                  featureMatrix, regressionMatrix, blasFactory_, lapackFactory_);
            }
    };

    /**
     * Allows to configure a rule learner that makes use of gradient boosting.
     */
    class BoostedRuleLearnerConfig : public RuleLearnerConfig,
                                     virtual public IBoostedRuleLearnerConfig {
        protected:

            /**
             * An unique pointer that stores the configuration of the rule heads.
             */
            std::unique_ptr<IHeadConfig> headConfigPtr_;

            /**
             * An unique pointer that stores the configuration of the data type that should be used for representing
             * gradients and Hessians.
             */
            std::unique_ptr<IStatisticTypeConfig> statisticTypeConfigPtr_;

            /**
             * A shared pointer that stores the configuration of the statistics that should be use in classification
             * problems.
             */
            std::shared_ptr<IClassificationStatisticsConfig> classificationStatisticsConfigPtr_;

            /**
             * A shared pointer that stores the configuration of the statistics that should be use in regression
             * problems.
             */
            std::shared_ptr<IRegressionStatisticsConfig> regressionStatisticsConfigPtr_;

            /**
             * A shared pointer that stores the configuration of the loss function that should be used in classification
             * problems.
             */
            std::shared_ptr<IClassificationLossConfig> classificationLossConfigPtr_;

            /**
             * A shared pointer that stores the configuration of the loss function that should be used in regression
             * problems.
             */
            std::shared_ptr<IRegressionLossConfig> regressionLossConfigPtr_;

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

        public:

            BoostedRuleLearnerConfig() : RuleLearnerConfig(BOOSTED_RULE_COMPARE_FUNCTION) {}

            virtual ~BoostedRuleLearnerConfig() override {}

            Property<IHeadConfig> getHeadConfig() override final {
                return util::property(headConfigPtr_);
            }

            Property<IStatisticTypeConfig> getStatisticTypeConfig() override final {
                return util::property(statisticTypeConfigPtr_);
            }

            ReadableProperty<IStatisticsConfig> getStatisticsConfig() const override final {
                return util::readableProperty<IStatisticsConfig, IClassificationStatisticsConfig,
                                              IRegressionStatisticsConfig>(classificationStatisticsConfigPtr_,
                                                                           regressionStatisticsConfigPtr_);
            }

            SharedProperty<IClassificationStatisticsConfig> getClassificationStatisticsConfig() override final {
                return util::sharedProperty(classificationStatisticsConfigPtr_);
            }

            SharedProperty<IRegressionStatisticsConfig> getRegressionStatisticsConfig() override final {
                return util::sharedProperty(regressionStatisticsConfigPtr_);
            }

            Property<IRegularizationConfig> getL1RegularizationConfig() override final {
                return util::property(l1RegularizationConfigPtr_);
            }

            Property<IRegularizationConfig> getL2RegularizationConfig() override final {
                return util::property(l2RegularizationConfigPtr_);
            }

            ReadableProperty<ILossConfig> getLossConfig() const override final {
                return util::readableProperty<ILossConfig, IClassificationLossConfig, IRegressionLossConfig>(
                  classificationLossConfigPtr_, regressionLossConfigPtr_);
            }

            SharedProperty<IClassificationLossConfig> getClassificationLossConfig() override final {
                return util::sharedProperty(classificationLossConfigPtr_);
            }

            SharedProperty<IRegressionLossConfig> getRegressionLossConfig() override final {
                return util::sharedProperty(regressionLossConfigPtr_);
            }

            Property<ILabelBinningConfig> getLabelBinningConfig() override final {
                return util::property(labelBinningConfigPtr_);
            }
    };
}
