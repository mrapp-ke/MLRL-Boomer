#include "boosting/multi_threading/parallel_statistic_update_auto.hpp"
#include "common/util/threads.hpp"


namespace boosting {

    AutoParallelStatisticUpdateConfig::AutoParallelStatisticUpdateConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : lossConfigPtr_(lossConfigPtr) {

    }

    uint32 AutoParallelStatisticUpdateConfig::configure(const IFeatureMatrix& featureMatrix,
                                                        const ILabelMatrix& labelMatrix) const {
        if (lossConfigPtr_->isDecomposable() || labelMatrix.getNumCols() < 20) {
            return getNumAvailableThreads(0);
        } else {
            return 0;
        }
    };

}
