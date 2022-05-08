#include "boosting/statistics/statistic_format_auto.hpp"


namespace boosting {

    AutomaticStatisticsConfig::AutomaticStatisticsConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr, const std::unique_ptr<IHeadConfig>& headConfigPtr,
            const std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr)
        : lossConfigPtr_(lossConfigPtr), headConfigPtr_(headConfigPtr), defaultRuleConfigPtr_(defaultRuleConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> AutomaticStatisticsConfig::createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
                const Lapack& lapack) const {
        bool preferSparseStatistics = labelMatrix.isSparse() && labelMatrix.getNumCols() > 1
                                      && !defaultRuleConfigPtr_->isDefaultRuleUsed() && headConfigPtr_->isPartial();
        return lossConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack,
                                                               preferSparseStatistics);
    }

}
