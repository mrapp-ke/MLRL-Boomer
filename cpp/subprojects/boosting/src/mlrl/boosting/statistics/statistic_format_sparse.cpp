#include "mlrl/boosting/statistics/statistic_format_sparse.hpp"

namespace boosting {

    SparseStatisticsConfig::SparseStatisticsConfig(ReadableProperty<ILossConfig> lossConfigGetter)
        : lossConfig_(lossConfigGetter) {}

    std::unique_ptr<IStatisticsProviderFactory> SparseStatisticsConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack) const {
        return lossConfig_.get().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack, true);
    }

    bool SparseStatisticsConfig::isDense() const {
        return false;
    }

    bool SparseStatisticsConfig::isSparse() const {
        return true;
    }

}
