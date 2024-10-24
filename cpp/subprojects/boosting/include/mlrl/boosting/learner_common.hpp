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

            const Blas blas_;

            const Lapack lapack_;

        public:

            /**
             * @param configPtr     An unique pointer to an object of type `IBoostedRuleLearnerConfig`
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
             */
            BoostedRuleLearnerConfigurator(std::unique_ptr<IBoostedRuleLearnerConfig> configPtr,
                                           Blas::DdotFunction ddotFunction, Blas::DspmvFunction dspmvFunction,
                                           Lapack::DsysvFunction dsysvFunction)
                : RuleLearnerConfigurator(*configPtr), configPtr_(std::move(configPtr)),
                  blas_(ddotFunction, dspmvFunction), lapack_(dsysvFunction) {}

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
                return configPtr_->getClassificationStatisticsConfig()
                  .get()
                  .createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, blas_, lapack_);
            }

            /**
             * @see `RuleLearnerConfigurator::createRegressionStatisticsProviderFactory`
             */
            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix) const override {
                return configPtr_->getRegressionStatisticsConfig().get().createRegressionStatisticsProviderFactory(
                  featureMatrix, regressionMatrix, blas_, lapack_);
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
