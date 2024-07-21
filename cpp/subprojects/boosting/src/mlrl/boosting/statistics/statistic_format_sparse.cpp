#include "mlrl/boosting/statistics/statistic_format_sparse.hpp"

namespace boosting {

    SparseClassificationStatisticsConfig::SparseClassificationStatisticsConfig(
      GetterFunction<IClassificationLossConfig> lossConfigGetter)
        : lossConfigGetter_(lossConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      SparseClassificationStatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
        const Lapack& lapack) const {
        return lossConfigGetter_().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack, true);
    }

    bool SparseClassificationStatisticsConfig::isDense() const {
        return false;
    }

    bool SparseClassificationStatisticsConfig::isSparse() const {
        return true;
    }

    SparseRegressionStatisticsConfig::SparseRegressionStatisticsConfig(
      GetterFunction<IRegressionLossConfig> lossConfigGetter)
        : lossConfigGetter_(lossConfigGetter) {}

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      SparseRegressionStatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
        const Lapack& lapack) const {
        return lossConfigGetter_().createStatisticsProviderFactory(featureMatrix, regressionMatrix, blas, lapack, true);
    }

    bool SparseRegressionStatisticsConfig::isDense() const {
        return false;
    }

    bool SparseRegressionStatisticsConfig::isSparse() const {
        return true;
    }

}
