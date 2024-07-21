#include "mlrl/boosting/statistics/statistic_format_auto.hpp"

namespace boosting {

    AutomaticClassificationStatisticsConfig::AutomaticClassificationStatisticsConfig(
      ReadableProperty<IClassificationLossConfig> lossConfigGetter, ReadableProperty<IHeadConfig> headConfigGetter,
      ReadableProperty<IDefaultRuleConfig> defaultRuleConfigGetter)
        : lossConfigGetter_(lossConfigGetter), headConfig_(headConfigGetter),
          defaultRuleConfig_(defaultRuleConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      AutomaticClassificationStatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
        const Lapack& lapack) const {
        bool defaultRuleUsed = defaultRuleConfig_.get().isDefaultRuleUsed(labelMatrix);
        bool partialHeadsUsed = headConfig_.get().isPartial();
        bool preferSparseStatistics = shouldSparseStatisticsBePreferred(labelMatrix, defaultRuleUsed, partialHeadsUsed);
        return lossConfigGetter_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, blas,
                                                                                     lapack, preferSparseStatistics);
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
        bool defaultRuleUsed = defaultRuleConfigGetter_.get().isDefaultRuleUsed(regressionMatrix);
        bool partialHeadsUsed = headConfigGetter_.get().isPartial();
        bool preferSparseStatistics =
          shouldSparseStatisticsBePreferred(regressionMatrix, defaultRuleUsed, partialHeadsUsed);
        return lossConfigGetter_.get().createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix, blas,
                                                                                 lapack, preferSparseStatistics);
    }

    AutomaticRegressionStatisticsConfig::AutomaticRegressionStatisticsConfig(
      ReadableProperty<IRegressionLossConfig> lossConfigGetter, ReadableProperty<IHeadConfig> headConfigGetter,
      ReadableProperty<IDefaultRuleConfig> defaultRuleConfigGetter)
        : lossConfigGetter_(lossConfigGetter), headConfigGetter_(headConfigGetter),
          defaultRuleConfigGetter_(defaultRuleConfigGetter) {}

    bool AutomaticRegressionStatisticsConfig::isDense() const {
        return false;
    }

    bool AutomaticRegressionStatisticsConfig::isSparse() const {
        return false;
    }

}
