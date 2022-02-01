#include "boosting/statistics/statistic_format_auto.hpp"


namespace boosting {

    AutomaticStatisticsConfig::AutomaticStatisticsConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : lossConfigPtr_(lossConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> AutomaticStatisticsConfig::createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix, const Blas& blas,
                const Lapack& lapack) const {
        bool preferSparseStatistics = false; // TODO Use correct value
        return lossConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack,
                                                               preferSparseStatistics);
    }

}
