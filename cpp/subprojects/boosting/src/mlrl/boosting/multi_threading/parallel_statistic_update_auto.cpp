#include "mlrl/boosting/multi_threading/parallel_statistic_update_auto.hpp"

#include "mlrl/common/util/threads.hpp"

namespace boosting {

    AutoParallelStatisticUpdateConfig::AutoParallelStatisticUpdateConfig(ReadableProperty<ILossConfig> lossConfig)
        : lossConfig_(lossConfig) {}

    MultiThreadingSettings AutoParallelStatisticUpdateConfig::getSettings(const IFeatureMatrix& featureMatrix,
                                                                          uint32 numOutputs) const {
        uint32 numThreads;

        if (!lossConfig_.get().isDecomposable() && numOutputs >= 20) {
            numThreads = util::getNumAvailableThreads(0);
        } else {
            numThreads = 1;
        }

        return MultiThreadingSettings(numThreads);
    }

}
