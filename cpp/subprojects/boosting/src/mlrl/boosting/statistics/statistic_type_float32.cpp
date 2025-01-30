#include "mlrl/boosting/statistics/statistic_type_float32.hpp"

namespace boosting {

    Float32StatisticsConfig::Float32StatisticsConfig(
      ReadableProperty<IClassificationStatisticsConfig> classificationStatisticsConfig,
      ReadableProperty<IRegressionStatisticsConfig> regressionStatisticsConfig)
        : classificationStatisticsConfig_(classificationStatisticsConfig),
          regressionStatisticsConfig_(regressionStatisticsConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      Float32StatisticsConfig::createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                             const IRowWiseLabelMatrix& labelMatrix,
                                                                             const BlasFactory& blasFactory,
                                                                             const LapackFactory& lapackFactory) const {
        return classificationStatisticsConfig_.get().createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, blasFactory, lapackFactory);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      Float32StatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        return regressionStatisticsConfig_.get().createRegressionStatisticsProviderFactory(
          featureMatrix, regressionMatrix, blasFactory, lapackFactory);
    }

}
