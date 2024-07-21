#include "mlrl/boosting/statistics/statistic_format_dense.hpp"

namespace boosting {

    DenseStatisticsConfig::DenseStatisticsConfig(
      GetterFunction<IClassificationLossConfig> classificationLossConfigGetter,
      GetterFunction<IRegressionLossConfig> regressionLossConfigGetter)
        : classificationLossConfigGetter_(classificationLossConfigGetter),
          regressionLossConfigGetter_(regressionLossConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      DenseStatisticsConfig::createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                           const IRowWiseLabelMatrix& labelMatrix,
                                                                           const Blas& blas,
                                                                           const Lapack& lapack) const {
        return classificationLossConfigGetter_().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas,
                                                                                 lapack, false);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      DenseStatisticsConfig::createRegressionStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                       const IRowWiseRegressionMatrix& regressionMatrix,
                                                                       const Blas& blas, const Lapack& lapack) const {
        return regressionLossConfigGetter_().createStatisticsProviderFactory(featureMatrix, regressionMatrix, blas,
                                                                             lapack, false);
    }

    bool DenseStatisticsConfig::isDense() const {
        return true;
    }

    bool DenseStatisticsConfig::isSparse() const {
        return false;
    }

}
