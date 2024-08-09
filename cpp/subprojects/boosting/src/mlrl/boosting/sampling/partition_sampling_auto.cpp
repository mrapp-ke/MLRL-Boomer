#include "mlrl/boosting/sampling/partition_sampling_auto.hpp"

#include "mlrl/common/sampling/partition_sampling_bi_random.hpp"
#include "mlrl/common/sampling/partition_sampling_bi_stratified_output_wise.hpp"
#include "mlrl/common/sampling/partition_sampling_no.hpp"

namespace boosting {

    static inline bool shouldUseHoldoutSet(
      const IGlobalPruningConfig& globalPruningConfig,
      const IMarginalProbabilityCalibratorConfig& marginalProbabilityCalibratorConfig,
      const IJointProbabilityCalibratorConfig& jointProbabilityCalibratorConfig) {
        return globalPruningConfig.shouldUseHoldoutSet() || marginalProbabilityCalibratorConfig.shouldUseHoldoutSet()
               || jointProbabilityCalibratorConfig.shouldUseHoldoutSet();
    }

    AutomaticPartitionSamplingConfig::AutomaticPartitionSamplingConfig(
      ReadableProperty<IGlobalPruningConfig> globalPruningConfig,
      ReadableProperty<IMarginalProbabilityCalibratorConfig> marginalProbabilityCalibratorConfig,
      ReadableProperty<IJointProbabilityCalibratorConfig> jointProbabilityCalibratorConfig)
        : globalPruningConfig_(globalPruningConfig),
          marginalProbabilityCalibratorConfig_(marginalProbabilityCalibratorConfig),
          jointProbabilityCalibratorConfig_(jointProbabilityCalibratorConfig) {}

    std::unique_ptr<IClassificationPartitionSamplingFactory>
      AutomaticPartitionSamplingConfig::createClassificationPartitionSamplingFactory() const {
        if (shouldUseHoldoutSet(globalPruningConfig_.get(), marginalProbabilityCalibratorConfig_.get(),
                                jointProbabilityCalibratorConfig_.get())) {
            return OutputWiseStratifiedBiPartitionSamplingConfig().createClassificationPartitionSamplingFactory();
        }

        return NoPartitionSamplingConfig().createClassificationPartitionSamplingFactory();
    }

    std::unique_ptr<IRegressionPartitionSamplingFactory>
      AutomaticPartitionSamplingConfig::createRegressionPartitionSamplingFactory() const {
        if (shouldUseHoldoutSet(globalPruningConfig_.get(), marginalProbabilityCalibratorConfig_.get(),
                                jointProbabilityCalibratorConfig_.get())) {
            return RandomBiPartitionSamplingConfig().createRegressionPartitionSamplingFactory();
        }

        return NoPartitionSamplingConfig().createRegressionPartitionSamplingFactory();
    }

}
