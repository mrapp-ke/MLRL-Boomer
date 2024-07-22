#include "mlrl/boosting/statistics/statistic_format_sparse.hpp"

namespace boosting {

    SparseStatisticsConfig::SparseStatisticsConfig(ReadableProperty<IClassificationLossConfig> classificationLossConfig,
                                                   ReadableProperty<IRegressionLossConfig> regressionLossConfig)
        : classificationLossConfig_(classificationLossConfig), regressionLossConfig_(regressionLossConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      SparseStatisticsConfig::createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                            const IRowWiseLabelMatrix& labelMatrix,
                                                                            const Blas& blas,
                                                                            const Lapack& lapack) const {
        return classificationLossConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix,
                                                                                             blas, lapack, true);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      SparseStatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
        const Lapack& lapack) const {
        return regressionLossConfig_.get().createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix,
                                                                                     blas, lapack, true);
    }

    bool SparseStatisticsConfig::isDense() const {
        return false;
    }

    bool SparseStatisticsConfig::isSparse() const {
        return true;
    }

}
