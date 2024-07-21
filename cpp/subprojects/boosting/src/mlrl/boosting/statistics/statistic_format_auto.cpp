#include "mlrl/boosting/statistics/statistic_format_auto.hpp"

namespace boosting {

    AutomaticClassificationStatisticsConfig::AutomaticClassificationStatisticsConfig(
      ReadableProperty<IClassificationLossConfig> lossConfig, ReadableProperty<IHeadConfig> headConfig,
      ReadableProperty<IDefaultRuleConfig> defaultRuleConfig)
        : lossConfig_(lossConfig), headConfig_(headConfig), defaultRuleConfig_(defaultRuleConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      AutomaticClassificationStatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
        const Lapack& lapack) const {
        bool defaultRuleUsed = defaultRuleConfig_.get().isDefaultRuleUsed(labelMatrix);
        bool partialHeadsUsed = headConfig_.get().isPartial();
        bool preferSparseStatistics = shouldSparseStatisticsBePreferred(labelMatrix, defaultRuleUsed, partialHeadsUsed);
        return lossConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack,
                                                                               preferSparseStatistics);
    }

    bool AutomaticClassificationStatisticsConfig::isDense() const {
        return false;
    }

    bool AutomaticClassificationStatisticsConfig::isSparse() const {
        return false;
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      AutomaticRegressionStatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
        const Lapack& lapack) const {
        bool defaultRuleUsed = defaultRuleConfig_.get().isDefaultRuleUsed(regressionMatrix);
        bool partialHeadsUsed = headConfig_.get().isPartial();
        bool preferSparseStatistics =
          shouldSparseStatisticsBePreferred(regressionMatrix, defaultRuleUsed, partialHeadsUsed);
        return lossConfig_.get().createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix, blas,
                                                                           lapack, preferSparseStatistics);
    }

    AutomaticRegressionStatisticsConfig::AutomaticRegressionStatisticsConfig(
      ReadableProperty<IRegressionLossConfig> lossConfig, ReadableProperty<IHeadConfig> headConfig,
      ReadableProperty<IDefaultRuleConfig> defaultRuleConfig)
        : lossConfig_(lossConfig), headConfig_(headConfig), defaultRuleConfig_(defaultRuleConfig) {}

    bool AutomaticRegressionStatisticsConfig::isDense() const {
        return false;
    }

    bool AutomaticRegressionStatisticsConfig::isSparse() const {
        return false;
    }

}
