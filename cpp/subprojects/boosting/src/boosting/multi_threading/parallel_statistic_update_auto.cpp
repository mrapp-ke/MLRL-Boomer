#include "boosting/multi_threading/parallel_statistic_update_auto.hpp"
#include "boosting/losses/loss_label_wise.hpp"
#include "common/util/threads.hpp"


namespace boosting {

    AutoParallelStatisticUpdateConfig::AutoParallelStatisticUpdateConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : lossConfigPtr_(lossConfigPtr) {

    }

    uint32 AutoParallelStatisticUpdateConfig::getNumThreads(const IFeatureMatrix& featureMatrix,
                                                            const ILabelMatrix& labelMatrix) const {
        if (dynamic_cast<const ILabelWiseLossConfig*>(lossConfigPtr_.get()) || labelMatrix.getNumCols() < 20) {
            return getNumAvailableThreads(0);
        } else {
            return 0;
        }
    };

}
