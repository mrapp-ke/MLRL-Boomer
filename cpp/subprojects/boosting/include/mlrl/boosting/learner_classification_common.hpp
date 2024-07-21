/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/learner_common.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * Allows to configure the individual modules of a rule learner that makes use of gradient boosting, depending on an
     * `IBoostedRuleLearnerConfig`.
     */
    class BoostedClassificationRuleLearnerConfigurator final : public BoostedRuleLearnerConfigurator {
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
            BoostedClassificationRuleLearnerConfigurator(std::unique_ptr<IBoostedRuleLearnerConfig> configPtr,
                                                         Blas::DdotFunction ddotFunction,
                                                         Blas::DspmvFunction dspmvFunction,
                                                         Lapack::DsysvFunction dsysvFunction)
                : BoostedRuleLearnerConfigurator(*configPtr), configPtr_(std::move(configPtr)),
                  blas_(ddotFunction, dspmvFunction), lapack_(dsysvFunction) {}

            /**
             * @see `RuleLearnerConfigurator::createStatisticsProviderFactory`
             */
            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override {
                return configPtr_->getClassificationStatisticsConfig()
                  .get()
                  .createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, blas_, lapack_);
            }
    };
}
