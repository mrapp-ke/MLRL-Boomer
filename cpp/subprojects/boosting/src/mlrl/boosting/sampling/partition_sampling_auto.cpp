#include "mlrl/boosting/sampling/partition_sampling_auto.hpp"

#include "mlrl/common/sampling/partition_sampling_bi_stratified_output_wise.hpp"
#include "mlrl/common/sampling/partition_sampling_no.hpp"

namespace boosting {

    AutomaticPartitionSamplingConfig::AutomaticPartitionSamplingConfig(
      ReadableProperty<IGlobalPruningConfig> globalPruningConfigGetter,
      ReadableProperty<IMarginalProbabilityCalibratorConfig> marginalProbabilityCalibratorConfigGetter,
      ReadableProperty<IJointProbabilityCalibratorConfig> jointProbabilityCalibratorConfigGetter)
        : globalPruningConfig_(globalPruningConfigGetter),
          marginalProbabilityCalibratorConfig_(marginalProbabilityCalibratorConfigGetter),
          jointProbabilityCalibratorConfig_(jointProbabilityCalibratorConfigGetter) {}

    std::unique_ptr<IClassificationPartitionSamplingFactory>
      AutomaticPartitionSamplingConfig::createClassificationPartitionSamplingFactory() const {
        if ((globalPruningConfig_.get().shouldUseHoldoutSet())
            || marginalProbabilityCalibratorConfig_.get().shouldUseHoldoutSet()
            || jointProbabilityCalibratorConfig_.get().shouldUseHoldoutSet()) {
            return OutputWiseStratifiedBiPartitionSamplingConfig().createClassificationPartitionSamplingFactory();
        }

        return NoPartitionSamplingConfig().createClassificationPartitionSamplingFactory();
    }

}
