#include "mlrl/boosting/statistics/statistic_format_auto.hpp"

namespace boosting {

    AutomaticStatisticsConfig::AutomaticStatisticsConfig(
      ReadableProperty<IClassificationLossConfig> classificationLossConfig,
      ReadableProperty<IRegressionLossConfig> regressionLossConfig, ReadableProperty<IHeadConfig> headConfig,
      ReadableProperty<IDefaultRuleConfig> defaultRuleConfig)
        : classificationLossConfig_(classificationLossConfig), regressionLossConfig_(regressionLossConfig),
          headConfig_(headConfig), defaultRuleConfig_(defaultRuleConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      AutomaticStatisticsConfig::createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                               const IRowWiseLabelMatrix& labelMatrix,
                                                                               const Blas& blas,
                                                                               const Lapack& lapack) const {
        bool defaultRuleUsed = defaultRuleConfig_.get().isDefaultRuleUsed(labelMatrix);
        bool partialHeadsUsed = headConfig_.get().isPartial();
        bool preferSparseStatistics = shouldSparseStatisticsBePreferred(labelMatrix, defaultRuleUsed, partialHeadsUsed);
        return classificationLossConfig_.get().createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, blas, lapack, preferSparseStatistics);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      AutomaticStatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
        const Lapack& lapack) const {
        bool defaultRuleUsed = defaultRuleConfig_.get().isDefaultRuleUsed(regressionMatrix);
        bool partialHeadsUsed = headConfig_.get().isPartial();
        bool preferSparseStatistics =
          shouldSparseStatisticsBePreferred(regressionMatrix, defaultRuleUsed, partialHeadsUsed);
        return regressionLossConfig_.get().createRegressionStatisticsProviderFactory(
          featureMatrix, regressionMatrix, blas, lapack, preferSparseStatistics);
    }

    bool AutomaticStatisticsConfig::isDense() const {
        return false;
    }

    bool AutomaticStatisticsConfig::isSparse() const {
        return false;
    }

}
