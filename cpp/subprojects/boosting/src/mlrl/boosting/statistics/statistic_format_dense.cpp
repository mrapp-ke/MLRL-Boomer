#include "mlrl/boosting/statistics/statistic_format_dense.hpp"

namespace boosting {

    DenseStatisticsConfig::DenseStatisticsConfig(ReadableProperty<IClassificationLossConfig> lossConfigGetter)
        : lossConfig_(lossConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory> DenseStatisticsConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack) const {
        return lossConfig_.get().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack, false);
    }

    bool DenseStatisticsConfig::isDense() const {
        return true;
    }

    bool DenseStatisticsConfig::isSparse() const {
        return false;
    }

}
