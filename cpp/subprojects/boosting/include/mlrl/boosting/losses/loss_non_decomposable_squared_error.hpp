/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_non_decomposable.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a loss function that implements a multivariate variant of the squared error loss that is
     * non-decomposable.
     */
    class NonDecomposableSquaredErrorLossConfig final : public INonDecomposableClassificationLossConfig,
                                                        public INonDecomposableRegressionLossConfig {
        private:

            const ReadableProperty<IHeadConfig> headConfig_;

        public:

            /**
             * @param headConfig A `ReadableProperty` that allows to access the `IHeadConfig` that stores the
             *                   configuration of rule heads
             */
            NonDecomposableSquaredErrorLossConfig(ReadableProperty<IHeadConfig> headConfig);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory,
              bool preferSparseStatistics) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory,
              bool preferSparseStatistics) const override;

            std::unique_ptr<IMarginalProbabilityFunctionFactory> createMarginalProbabilityFunctionFactory()
              const override;

            std::unique_ptr<IJointProbabilityFunctionFactory> createJointProbabilityFunctionFactory() const override;

            float64 getDefaultPrediction() const override;

            std::unique_ptr<INonDecomposableClassificationLossFactory> createNonDecomposableClassificationLossFactory()
              const override;

            std::unique_ptr<INonDecomposableRegressionLossFactory> createNonDecomposableRegressionLossFactory()
              const override;
    };

}
