#include "mlrl/boosting/statistics/statistic_format_dense.hpp"

namespace boosting {

    DenseStatisticsConfig::DenseStatisticsConfig(GetterFunction<ILossConfig> lossConfigGetter)
        : lossConfigGetter_(lossConfigGetter) {}

    std::unique_ptr<IStatisticsProviderFactory> DenseStatisticsConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack) const {
        return lossConfigGetter_().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack, false);
    }

    bool DenseStatisticsConfig::isDense() const {
        return true;
    }

    bool DenseStatisticsConfig::isSparse() const {
        return false;
    }

}
