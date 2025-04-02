/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/boosting/statistics/statistic_type.hpp"
#include "mlrl/common/util/properties.hpp"

namespace boosting {

    /**
     * Allows to use 64-bit floating point values for representing statistics about the quality of predictions for
     * training examples.
     */
    class Float64StatisticsConfig final : public IStatisticTypeConfig {
        private:

            const ReadableProperty<IHeadConfig> headConfig_;

        public:

            /**
             * @param headConfig A `ReadableProperty` that allows to access the `IHeadConfig` that stores the
             *                   configuration of rule heads
             */
            Float64StatisticsConfig(ReadableProperty<IHeadConfig> headConfig);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const IDecomposableClassificationLossConfig& lossConfig) const override;

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const ISparseDecomposableClassificationLossConfig& lossConfig) const override;

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const INonDecomposableClassificationLossConfig& lossConfig, const BlasFactory& blasFactory,
              const LapackFactory& lapackFactory) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              const IDecomposableRegressionLossConfig& lossConfig) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              const INonDecomposableRegressionLossConfig& lossConfig, const BlasFactory& blasFactory,
              const LapackFactory& lapackFactory) const override;
    };

}
