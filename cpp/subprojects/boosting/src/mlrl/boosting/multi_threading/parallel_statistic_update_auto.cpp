#include "mlrl/boosting/multi_threading/parallel_statistic_update_auto.hpp"

#include "mlrl/common/util/threads.hpp"

namespace boosting {

    AutoParallelStatisticUpdateConfig::AutoParallelStatisticUpdateConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : lossConfigPtr_(lossConfigPtr) {}

    uint32 AutoParallelStatisticUpdateConfig::getNumThreads(const IFeatureMatrix& featureMatrix,
                                                            uint32 numOutputs) const {
        if (!lossConfigPtr_->isDecomposable() && numOutputs >= 20) {
            return getNumAvailableThreads(0);
        } else {
            return 1;
        }
    }

}
