#include "mlrl/boosting/statistics/statistic_format_auto.hpp"

namespace boosting {

    AutomaticStatisticsConfig::AutomaticStatisticsConfig(
      ReadableProperty<IClassificationLossConfig> classificationLossConfig,
      ReadableProperty<IRegressionLossConfig> regressionLossConfigGetter, GetterFunction<IHeadConfig> headConfig,
      ReadableProperty<IDefaultRuleConfig> defaultRuleConfigGetter)
        : classificationLossConfigGetter_(classificationLossConfigGetter),
          regressionLossConfigGetter_(regressionLossConfigGetter), headConfigGetter_(headConfigGetter),
          defaultRuleConfigGetter_(defaultRuleConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      AutomaticStatisticsConfig::createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                               const IRowWiseLabelMatrix& labelMatrix,
                                                                               const Blas& blas,
                                                                               const Lapack& lapack) const {
        bool defaultRuleUsed = defaultRuleConfigGetter_().isDefaultRuleUsed(labelMatrix);
        bool partialHeadsUsed = headConfigGetter_().isPartial();
        bool preferSparseStatistics = shouldSparseStatisticsBePreferred(labelMatrix, defaultRuleUsed, partialHeadsUsed);
        return classificationLossConfigGetter_().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas,
                                                                                 lapack, preferSparseStatistics);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      AutomaticStatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
        const Lapack& lapack) const {
        bool defaultRuleUsed = defaultRuleConfigGetter_().isDefaultRuleUsed(regressionMatrix);
        bool partialHeadsUsed = headConfigGetter_().isPartial();
        bool preferSparseStatistics =
          shouldSparseStatisticsBePreferred(regressionMatrix, defaultRuleUsed, partialHeadsUsed);
        return regressionLossConfigGetter_().createStatisticsProviderFactory(featureMatrix, regressionMatrix, blas,
                                                                             lapack, preferSparseStatistics);
    }

    bool AutomaticStatisticsConfig::isDense() const {
        return false;
    }

    bool AutomaticStatisticsConfig::isSparse() const {
        return false;
    }

}
