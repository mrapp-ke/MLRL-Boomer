#include "mlrl/boosting/sampling/partition_sampling_auto.hpp"

#include "mlrl/common/sampling/partition_sampling_bi_stratified_output_wise.hpp"
#include "mlrl/common/sampling/partition_sampling_no.hpp"

namespace boosting {

    AutomaticPartitionSamplingConfig::AutomaticPartitionSamplingConfig(
      GetterFunction<IGlobalPruningConfig> globalPruningConfigGetter,
      GetterFunction<IMarginalProbabilityCalibratorConfig> marginalProbabilityCalibratorConfigGetter,
      GetterFunction<IJointProbabilityCalibratorConfig> jointProbabilityCalibratorConfigGetter)
        : globalPruningConfigGetter_(globalPruningConfigGetter),
          marginalProbabilityCalibratorConfigGetter_(marginalProbabilityCalibratorConfigGetter),
          jointProbabilityCalibratorConfigGetter_(jointProbabilityCalibratorConfigGetter) {}

    std::unique_ptr<IPartitionSamplingFactory> AutomaticPartitionSamplingConfig::createPartitionSamplingFactory()
      const {
        if ((globalPruningConfigGetter_().shouldUseHoldoutSet())
            || marginalProbabilityCalibratorConfigGetter_().shouldUseHoldoutSet()
            || jointProbabilityCalibratorConfigGetter_().shouldUseHoldoutSet()) {
            return OutputWiseStratifiedBiPartitionSamplingConfig().createPartitionSamplingFactory();
        }

        return NoPartitionSamplingConfig().createPartitionSamplingFactory();
    }

}
