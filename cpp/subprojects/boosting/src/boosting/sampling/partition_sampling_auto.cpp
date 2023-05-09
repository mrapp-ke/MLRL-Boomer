#include "boosting/sampling/partition_sampling_auto.hpp"

#include "common/sampling/partition_sampling_bi_random.hpp"
#include "common/sampling/partition_sampling_bi_stratified_label_wise.hpp"
#include "common/sampling/partition_sampling_no.hpp"

namespace boosting {

    AutomaticPartitionSamplingConfig::AutomaticPartitionSamplingConfig(
      const std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr,
      const std::unique_ptr<IMarginalProbabilityCalibratorConfig>& marginalProbabilityCalibratorConfigPtr,
      const std::unique_ptr<IJointProbabilityCalibratorConfig>& jointProbabilityCalibratorConfigPtr,
      const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : globalPruningConfigPtr_(globalPruningConfigPtr),
          marginalProbabilityCalibratorConfigPtr_(marginalProbabilityCalibratorConfigPtr),
          jointProbabilityCalibratorConfigPtr_(jointProbabilityCalibratorConfigPtr), lossConfigPtr_(lossConfigPtr) {}

    std::unique_ptr<IPartitionSamplingFactory> AutomaticPartitionSamplingConfig::createPartitionSamplingFactory()
      const {
        if ((globalPruningConfigPtr_.get() && globalPruningConfigPtr_->shouldUseHoldoutSet())
            || marginalProbabilityCalibratorConfigPtr_->shouldUseHoldoutSet()
            || jointProbabilityCalibratorConfigPtr_->shouldUseHoldoutSet()) {
            if (lossConfigPtr_->isDecomposable()) {
                return LabelWiseStratifiedBiPartitionSamplingConfig().createPartitionSamplingFactory();
            } else {
                return RandomBiPartitionSamplingConfig().createPartitionSamplingFactory();
            }
        }

        return NoPartitionSamplingConfig().createPartitionSamplingFactory();
    }

}
