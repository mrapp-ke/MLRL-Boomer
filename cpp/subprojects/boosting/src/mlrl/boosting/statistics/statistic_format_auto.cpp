#include "mlrl/boosting/statistics/statistic_format_auto.hpp"

namespace boosting {

    AutomaticStatisticsConfig::AutomaticStatisticsConfig(GetterFunction<ILossConfig> lossConfigGetter,
                                                         GetterFunction<IHeadConfig> headConfigGetter,
                                                         GetterFunction<IDefaultRuleConfig> defaultRuleConfigGetter)
        : lossConfigGetter_(lossConfigGetter), headConfigGetter_(headConfigGetter),
          defaultRuleConfigGetter_(defaultRuleConfigGetter) {}

    std::unique_ptr<IStatisticsProviderFactory> AutomaticStatisticsConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack) const {
        bool defaultRuleUsed = defaultRuleConfigGetter_().isDefaultRuleUsed(labelMatrix);
        bool partialHeadsUsed = headConfigGetter_().isPartial();
        bool preferSparseStatistics = shouldSparseStatisticsBePreferred(labelMatrix, defaultRuleUsed, partialHeadsUsed);
        return lossConfigGetter_().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack,
                                                                   preferSparseStatistics);
    }

    bool AutomaticStatisticsConfig::isDense() const {
        return false;
    }

    bool AutomaticStatisticsConfig::isSparse() const {
        return false;
    }

}
