/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

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
             * @see `RuleLearnerConfigurator::createStatisticsProviderFactory`
             */
            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override {
                return configPtr_->getStatisticsConfig().get().createStatisticsProviderFactory(
                  featureMatrix, labelMatrix, blas_, lapack_);
            }

            /**
             * @see `RuleLearnerConfigurator::createModelBuilderFactory`
             */
            std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const override {
                return std::make_unique<RuleListBuilderFactory>();
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
             * An unique pointer that stores the configuration of the statistics.
             */
            std::unique_ptr<IStatisticsConfig> statisticsConfigPtr_;

            /**
             * An unique pointer that stores the configuration of the loss function.
             */
            std::unique_ptr<IClassificationLossConfig> lossConfigPtr_;

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

            BoostedRuleLearnerConfig()
                : RuleLearnerConfig(BOOSTED_RULE_COMPARE_FUNCTION),
                  headConfigPtr_(std::make_unique<CompleteHeadConfig>(
                    readableProperty(labelBinningConfigPtr_), readableProperty(parallelStatisticUpdateConfigPtr_),
                    readableProperty(l1RegularizationConfigPtr_), readableProperty(l2RegularizationConfigPtr_))),
                  statisticsConfigPtr_(std::make_unique<DenseStatisticsConfig>(readableProperty(lossConfigPtr_))),
                  lossConfigPtr_(std::make_unique<DecomposableLogisticLossConfig>(readableProperty(headConfigPtr_))),
                  l1RegularizationConfigPtr_(std::make_unique<NoRegularizationConfig>()),
                  l2RegularizationConfigPtr_(std::make_unique<NoRegularizationConfig>()),
                  labelBinningConfigPtr_(std::make_unique<NoLabelBinningConfig>(
                    readableProperty(l1RegularizationConfigPtr_), readableProperty(l2RegularizationConfigPtr_))) {}

            virtual ~BoostedRuleLearnerConfig() override {}

            Property<IHeadConfig> getHeadConfig() override final {
                return property(headConfigPtr_);
            }

            Property<IStatisticsConfig> getStatisticsConfig() override final {
                return property(statisticsConfigPtr_);
            }

            Property<IRegularizationConfig> getL1RegularizationConfig() override final {
                return property(l1RegularizationConfigPtr_);
            }

            Property<IRegularizationConfig> getL2RegularizationConfig() override final {
                return property(l2RegularizationConfigPtr_);
            }

            Property<IClassificationLossConfig> getLossConfig() override final {
                return property(lossConfigPtr_);
            }

            Property<ILabelBinningConfig> getLabelBinningConfig() override final {
                return property(labelBinningConfigPtr_);
            }
    };
}

#ifdef _WIN32
    #pragma warning(pop)
#endif
