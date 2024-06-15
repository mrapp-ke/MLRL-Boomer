/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/learner.hpp"
#include "mlrl/boosting/model/rule_list_builder.hpp"
#include "mlrl/boosting/rule_evaluation/rule_compare_function.hpp"
#include "mlrl/common/learner_common.hpp"

namespace boosting {

    /**
     * Allows to configure the individual modules of a rule learner that makes use of gradient boosting, depending on an
     * `IBoostedRuleLearner::IConfig`.
     */
    class BoostedRuleLearnerConfigurator final : public RuleLearnerConfigurator {
        private:

            const std::unique_ptr<IBoostedRuleLearner::IConfig> configPtr_;

            const Blas blas_;

            const Lapack lapack_;

        public:

            /**
             * @param configPtr     An unique pointer to an object of type `IBoostedRuleLearner::IConfig`
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
             */
            BoostedRuleLearnerConfigurator(std::unique_ptr<IBoostedRuleLearner::IConfig> configPtr,
                                           Blas::DdotFunction ddotFunction, Blas::DspmvFunction dspmvFunction,
                                           Lapack::DsysvFunction dsysvFunction)
                : RuleLearnerConfigurator(*configPtr), configPtr_(std::move(configPtr)),
                  blas_(ddotFunction, dspmvFunction), lapack_(dsysvFunction) {}

            /**
             * @see `RuleLearnerConfigurator::createStatisticsProviderFactory`
             */
            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override {
                return configPtr_->getStatisticsConfigPtr()->createStatisticsProviderFactory(featureMatrix, labelMatrix,
                                                                                             blas_, lapack_);
            }

            /**
             * @see `RuleLearnerConfigurator::createModelBuilderFactory`
             */
            std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const override {
                return std::make_unique<RuleListBuilderFactory>();
            }
    };

    /**
     * An abstract base class for all rule learners that makes use of gradient boosting.
     */
    class AbstractBoostedRuleLearner : public AbstractRuleLearner,
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

                public:

                    Config()
                        : AbstractRuleLearner::Config(BOOSTED_RULE_COMPARE_FUNCTION),
                          headConfigPtr_(std::make_unique<CompleteHeadConfig>(
                            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_, l1RegularizationConfigPtr_,
                            l2RegularizationConfigPtr_)),
                          statisticsConfigPtr_(std::make_unique<DenseStatisticsConfig>(lossConfigPtr_)),
                          lossConfigPtr_(std::make_unique<DecomposableLogisticLossConfig>(headConfigPtr_)),
                          l1RegularizationConfigPtr_(std::make_unique<NoRegularizationConfig>()),
                          l2RegularizationConfigPtr_(std::make_unique<NoRegularizationConfig>()),
                          labelBinningConfigPtr_(std::make_unique<NoLabelBinningConfig>(l1RegularizationConfigPtr_,
                                                                                        l2RegularizationConfigPtr_)) {}

                    std::unique_ptr<IHeadConfig>& getHeadConfigPtr() override final {
                        return headConfigPtr_;
                    }

                    std::unique_ptr<IStatisticsConfig>& getStatisticsConfigPtr() override final {
                        return statisticsConfigPtr_;
                    }

                    std::unique_ptr<IRegularizationConfig>& getL1RegularizationConfigPtr() override final {
                        return l1RegularizationConfigPtr_;
                    }

                    std::unique_ptr<IRegularizationConfig>& getL2RegularizationConfigPtr() override final {
                        return l2RegularizationConfigPtr_;
                    }

                    std::unique_ptr<ILossConfig>& getLossConfigPtr() override final {
                        return lossConfigPtr_;
                    }

                    std::unique_ptr<ILabelBinningConfig>& getLabelBinningConfigPtr() override final {
                        return labelBinningConfigPtr_;
                    }
            };

            /**
             * @param configurator A reference to an object of type `BoostedRuleLearnerConfigurator` that allows to
             *                     configure the individual modules to be used by the rule learner
             */
            AbstractBoostedRuleLearner(const BoostedRuleLearnerConfigurator& configurator)
                : AbstractRuleLearner(configurator) {}
    };

}
