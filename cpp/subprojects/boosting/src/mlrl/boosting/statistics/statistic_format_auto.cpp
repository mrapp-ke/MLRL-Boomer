#include "mlrl/boosting/statistics/statistic_format_auto.hpp"

namespace boosting {

    AutomaticStatisticsConfig::AutomaticStatisticsConfig(ReadableProperty<ILossConfig> lossConfigGetter,
                                                         ReadableProperty<IHeadConfig> headConfigGetter,
                                                         ReadableProperty<IDefaultRuleConfig> defaultRuleConfigGetter)
        : lossConfig_(lossConfigGetter), headConfig_(headConfigGetter), defaultRuleConfig_(defaultRuleConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      AutomaticStatisticsConfig::createStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                 const IRowWiseLabelMatrix& labelMatrix,
                                                                 const Blas& blas, const Lapack& lapack) const {
        bool defaultRuleUsed = defaultRuleConfig_.get().isDefaultRuleUsed(labelMatrix);
        bool partialHeadsUsed = headConfig_.get().isPartial();
        bool preferSparseStatistics = shouldSparseStatisticsBePreferred(labelMatrix, defaultRuleUsed, partialHeadsUsed);
        return lossConfig_.get().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack,
                                                                 preferSparseStatistics);
    }

    bool AutomaticStatisticsConfig::isDense() const {
        return false;
    }

    bool AutomaticStatisticsConfig::isSparse() const {
        return false;
    }

}
