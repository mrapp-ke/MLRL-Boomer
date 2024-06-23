/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_non_decomposable.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a loss function that implements a multivariate variant of the squared error loss that is
     * non-decomposable.
     */
    class NonDecomposableSquaredErrorLossConfig final : public INonDecomposableLossConfig {
        private:

            const std::unique_ptr<IHeadConfig>& headConfigPtr_;

        public:

            /**
             * @param headConfigPtr A reference to an unique pointer that stores the configuration of rule heads
             */
            NonDecomposableSquaredErrorLossConfig(const std::unique_ptr<IHeadConfig>& headConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack, bool preferSparseStatistics) const override;

            std::unique_ptr<IMarginalProbabilityFunctionFactory> createMarginalProbabilityFunctionFactory()
              const override;

            std::unique_ptr<IJointProbabilityFunctionFactory> createJointProbabilityFunctionFactory() const override;

            float64 getDefaultPrediction() const override;

            std::unique_ptr<INonDecomposableLossFactory> createNonDecomposableLossFactory() const override;
    };

}
