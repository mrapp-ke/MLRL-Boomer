#include "mlrl/boosting/statistics/statistic_format_sparse.hpp"

namespace boosting {

    SparseStatisticsConfig::SparseStatisticsConfig(ReadableProperty<IClassificationLossConfig> classificationLossConfig,
                                                   ReadableProperty<IRegressionLossConfig> regressionLossConfig)
        : classificationLossConfig_(classificationLossConfig), regressionLossConfig_(regressionLossConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      SparseStatisticsConfig::createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                            const IRowWiseLabelMatrix& labelMatrix,
                                                                            const BlasFactory& blasFactory,
                                                                            const LapackFactory& lapackFactory) const {
        return classificationLossConfig_.get().createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, blasFactory, lapackFactory, true);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      SparseStatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        return regressionLossConfig_.get().createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix,
                                                                                     blasFactory, lapackFactory, true);
    }

    bool SparseStatisticsConfig::isDense() const {
        return false;
    }

    bool SparseStatisticsConfig::isSparse() const {
        return true;
    }

}
