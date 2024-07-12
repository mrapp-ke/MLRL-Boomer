#include "mlrl/boosting/multi_threading/parallel_rule_refinement_auto.hpp"

#include "mlrl/common/util/threads.hpp"

namespace boosting {

    AutoParallelRuleRefinementConfig::AutoParallelRuleRefinementConfig(
      GetterFunction<ILossConfig> lossConfigGetter, GetterFunction<IHeadConfig> headConfigGetter,
      GetterFunction<IFeatureSamplingConfig> featureSamplingConfigGetter)
        : lossConfigGetter_(lossConfigGetter), headConfigGetter_(headConfigGetter),
          featureSamplingConfigGetter_(featureSamplingConfigGetter) {}

    uint32 AutoParallelRuleRefinementConfig::getNumThreads(const IFeatureMatrix& featureMatrix,
                                                           uint32 numOutputs) const {
        if (!lossConfigGetter_().isDecomposable() && !headConfigGetter_().isSingleOutput()) {
            return 1;
        } else if (featureMatrix.isSparse() && !featureSamplingConfigGetter_().isSamplingUsed()) {
            return 1;
        } else {
            return getNumAvailableThreads(0);
        }
    }

}
