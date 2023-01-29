#include "boosting/sampling/partition_sampling_auto.hpp"

#include "common/sampling/partition_sampling_bi_random.hpp"
#include "common/sampling/partition_sampling_bi_stratified_label_wise.hpp"
#include "common/sampling/partition_sampling_no.hpp"

namespace boosting {

    AutomaticPartitionSamplingConfig::AutomaticPartitionSamplingConfig(
            const std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr,
            const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : globalPruningConfigPtr_(globalPruningConfigPtr), lossConfigPtr_(lossConfigPtr) {}

    std::unique_ptr<IPartitionSamplingFactory> AutomaticPartitionSamplingConfig::createPartitionSamplingFactory()
        const {
        if (globalPruningConfigPtr_.get() && globalPruningConfigPtr_->shouldUseHoldoutSet()) {
            if (lossConfigPtr_->isDecomposable()) {
                return LabelWiseStratifiedBiPartitionSamplingConfig().createPartitionSamplingFactory();
            } else {
                return RandomBiPartitionSamplingConfig().createPartitionSamplingFactory();
            }
        }

        return NoPartitionSamplingConfig().createPartitionSamplingFactory();
    }

}
