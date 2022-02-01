#include "boosting/statistics/statistic_format_auto.hpp"


namespace boosting {

    AutomaticStatisticsConfig::AutomaticStatisticsConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr, const std::unique_ptr<IHeadConfig>& headConfigPtr,
            const std::unique_ptr<IRuleModelAssemblageConfig>& ruleModelAssemblageConfigPtr)
        : lossConfigPtr_(lossConfigPtr), headConfigPtr_(headConfigPtr),
          ruleModelAssemblageConfigPtr_(ruleModelAssemblageConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> AutomaticStatisticsConfig::createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix, const Blas& blas,
                const Lapack& lapack) const {
        bool preferSparseStatistics =
            !ruleModelAssemblageConfigPtr_->isDefaultRuleUsed() && headConfigPtr_->isPartial();
        return lossConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack,
                                                               preferSparseStatistics);
    }

}
