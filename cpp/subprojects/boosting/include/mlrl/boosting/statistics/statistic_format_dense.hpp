/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/statistics/statistic_format.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a dense format for storing statistics about the quality of predictions for training examples.
     */
    class DenseStatisticsConfig final : public IClassificationStatisticsConfig,
                                        public IRegressionStatisticsConfig {
        private:

            const ReadableProperty<IClassificationLossConfig> classificationLossConfig_;

            const ReadableProperty<IRegressionLossConfig> regressionLossConfig_;

        public:

            /**
             * @param classificationLossConfig  A `ReadableProperty` that allows to access the
             *                                  `IClassificationLossConfig` that stores the configuration of the loss
             *                                  function that should be used in classification problems
             * @param regressionLossConfig      A `ReadableProperty` that allows to access the `IRegressionLossConfig`
             *                                  that stores the configuration of the loss function that should be used
             *                                  in classification problems
             */
            DenseStatisticsConfig(ReadableProperty<IClassificationLossConfig> classificationLossConfig,
                                  ReadableProperty<IRegressionLossConfig> regressionLossConfig);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

}
