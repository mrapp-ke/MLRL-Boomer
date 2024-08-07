#include "mlrl/boosting/statistics/statistic_format_dense.hpp"

namespace boosting {

    DenseStatisticsConfig::DenseStatisticsConfig(ReadableProperty<IClassificationLossConfig> classificationLossConfig,
                                                 ReadableProperty<IRegressionLossConfig> regressionLossConfig)
        : classificationLossConfig_(classificationLossConfig), regressionLossConfig_(regressionLossConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      DenseStatisticsConfig::createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                           const IRowWiseLabelMatrix& labelMatrix,
                                                                           const Blas& blas,
                                                                           const Lapack& lapack) const {
        return classificationLossConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix,
                                                                                             blas, lapack, false);
    }

    bool DenseStatisticsConfig::isDense() const {
        return true;
    }

    bool DenseStatisticsConfig::isSparse() const {
        return false;
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      DenseStatisticsConfig::createRegressionStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                       const IRowWiseRegressionMatrix& regressionMatrix,
                                                                       const Blas& blas, const Lapack& lapack) const {
        return regressionLossConfig_.get().createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix,
                                                                                     blas, lapack, false);
    }

}
