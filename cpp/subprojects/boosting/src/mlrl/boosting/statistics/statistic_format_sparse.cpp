#include "mlrl/boosting/statistics/statistic_format_sparse.hpp"

namespace boosting {

    SparseStatisticsConfig::SparseStatisticsConfig(
      GetterFunction<IClassificationLossConfig> classificationLossConfigGetter,
      GetterFunction<IRegressionLossConfig> regressionLossConfigGetter)
        : classificationLossConfigGetter_(classificationLossConfigGetter),
          regressionLossConfigGetter_(regressionLossConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      SparseStatisticsConfig::createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                            const IRowWiseLabelMatrix& labelMatrix,
                                                                            const Blas& blas,
                                                                            const Lapack& lapack) const {
        return classificationLossConfigGetter_().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas,
                                                                                 lapack, true);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      SparseStatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
        const Lapack& lapack) const {
        return regressionLossConfigGetter_().createStatisticsProviderFactory(featureMatrix, regressionMatrix, blas,
                                                                             lapack, true);
    }

    bool SparseStatisticsConfig::isDense() const {
        return false;
    }

    bool SparseStatisticsConfig::isSparse() const {
        return true;
    }

}
