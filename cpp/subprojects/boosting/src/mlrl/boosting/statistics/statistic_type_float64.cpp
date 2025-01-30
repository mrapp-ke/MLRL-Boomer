#include "mlrl/boosting/statistics/statistic_type_float64.hpp"

namespace boosting {

    Float64StatisticsConfig::Float64StatisticsConfig(
      ReadableProperty<IClassificationStatisticsConfig> classificationStatisticsConfig,
      ReadableProperty<IRegressionStatisticsConfig> regressionStatisticsConfig)
        : classificationStatisticsConfig_(classificationStatisticsConfig),
          regressionStatisticsConfig_(regressionStatisticsConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      Float64StatisticsConfig::createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                             const IRowWiseLabelMatrix& labelMatrix,
                                                                             const BlasFactory& blasFactory,
                                                                             const LapackFactory& lapackFactory) const {
        return classificationStatisticsConfig_.get().createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, blasFactory, lapackFactory);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      Float64StatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        return regressionStatisticsConfig_.get().createRegressionStatisticsProviderFactory(
          featureMatrix, regressionMatrix, blasFactory, lapackFactory);
    }

}
