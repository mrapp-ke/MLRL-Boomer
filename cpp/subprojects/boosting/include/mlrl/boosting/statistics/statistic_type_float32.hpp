/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistic_format.hpp"
#include "mlrl/boosting/statistics/statistic_type.hpp"
#include "mlrl/common/util/properties.hpp"

namespace boosting {

    /**
     * Allows to use 32-bit floating point values for representing statistics about the quality of predictions for
     * training examples.
     */
    class Float32StatisticsConfig final : public IStatisticTypeConfig {
        private:

            const ReadableProperty<IClassificationStatisticsConfig> classificationStatisticsConfig_;

            const ReadableProperty<IRegressionStatisticsConfig> regressionStatisticsConfig_;

        public:

            /**
             * @param classificationStatisticsConfig  A `ReadableProperty` that allows to access the
             *                                        `IClassificationStatisticsConfig` that stores the configuration of
             *                                        the format that should be used for storing statistics about the
             *                                        quality of predictions for training examples in classification
             *                                        problems
             * @param regressionStatisticsConfig      A `ReadableProperty` that allows to access the
             *                                        `IRegressionStatisticsConfig` that stores the configuration of
             *                                        the format that should be used for storing statistics about the
             *                                        quality of predictions for training examples in regression
             *                                        problems
             */
            Float32StatisticsConfig(ReadableProperty<IClassificationStatisticsConfig> classificationStatisticsConfig,
                                    ReadableProperty<IRegressionStatisticsConfig> regressionStatisticsConfig);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override;
    };

}
