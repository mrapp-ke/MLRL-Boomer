#include "mlrl/boosting/multi_threading/parallel_rule_refinement_auto.hpp"

#include "mlrl/common/util/threads.hpp"

namespace boosting {

    AutoParallelRuleRefinementConfig::AutoParallelRuleRefinementConfig(
      ReadableProperty<ILossConfig> lossConfig, ReadableProperty<IHeadConfig> headConfig,
      ReadableProperty<IFeatureSamplingConfig> featureSamplingConfig)
        : lossConfig_(lossConfig), headConfig_(headConfig), featureSamplingConfig_(featureSamplingConfig) {}

    uint32 AutoParallelRuleRefinementConfig::getNumThreads(const IFeatureMatrix& featureMatrix,
                                                           uint32 numOutputs) const {
        if (!lossConfig_.get().isDecomposable() && !headConfig_.get().isSingleOutput()) {
            return 1;
        } else if (featureMatrix.isSparse() && !featureSamplingConfig_.get().isSamplingUsed()) {
            return 1;
        } else {
            return getNumAvailableThreads(0);
        }
    }

}
