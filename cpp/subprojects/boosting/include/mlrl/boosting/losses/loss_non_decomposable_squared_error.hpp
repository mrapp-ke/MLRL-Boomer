/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_non_decomposable.hpp"
#include "mlrl/boosting/statistics/statistic_type.hpp"
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

            const ReadableProperty<IStatisticTypeConfig> statisticTypeConfig_;

        public:

            /**
             *  @param statisticTypeConfig  A `ReadableProperty` that allows to access the `IStatisticTypeConfig` that
             *                              stores the configuration of the data type that should be used for
             *                              representing statistics about the quality of predictions for training
             *                              examples
             */
            NonDecomposableSquaredErrorLossConfig(ReadableProperty<IStatisticTypeConfig> statisticTypeConfig);

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

            std::unique_ptr<INonDecomposableClassificationLossConfig::IPreset<float32>>
              createNonDecomposable32BitClassificationPreset() const override;

            std::unique_ptr<INonDecomposableClassificationLossConfig::IPreset<float64>>
              createNonDecomposable64BitClassificationPreset() const override;

            std::unique_ptr<INonDecomposableRegressionLossConfig::IPreset<float32>>
              createNonDecomposable32BitRegressionPreset() const override;

            std::unique_ptr<INonDecomposableRegressionLossConfig::IPreset<float64>>
              createNonDecomposable64BitRegressionPreset() const override;
    };

}
