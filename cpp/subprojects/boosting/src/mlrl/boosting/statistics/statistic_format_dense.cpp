#include "mlrl/boosting/statistics/statistic_format_dense.hpp"

namespace boosting {

    DenseClassificationStatisticsConfig::DenseClassificationStatisticsConfig(
      ReadableProperty<IClassificationLossConfig> lossConfig)
        : lossConfig_(lossConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      DenseClassificationStatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
        const Lapack& lapack) const {
        return lossConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack,
                                                                               false);
    }

    bool DenseClassificationStatisticsConfig::isDense() const {
        return true;
    }

    bool DenseClassificationStatisticsConfig::isSparse() const {
        return false;
    }

    DenseRegressionStatisticsConfig::DenseRegressionStatisticsConfig(ReadableProperty<IRegressionLossConfig> lossConfig)
        : lossConfig_(lossConfig) {}

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      DenseRegressionStatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
        const Lapack& lapack) const {
        return lossConfig_.get().createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix, blas,
                                                                           lapack, false);
    }

    bool DenseRegressionStatisticsConfig::isDense() const {
        return true;
    }

    bool DenseRegressionStatisticsConfig::isSparse() const {
        return false;
    }

}
