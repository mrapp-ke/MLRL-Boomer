#include "mlrl/boosting/multi_threading/parallel_rule_refinement_auto.hpp"

#include "mlrl/common/util/threads.hpp"

namespace boosting {

    AutoParallelRuleRefinementConfig::AutoParallelRuleRefinementConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr, const std::unique_ptr<IHeadConfig>& headConfigPtr,
      const std::unique_ptr<IFeatureSamplingConfig>& featureSamplingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), headConfigPtr_(headConfigPtr),
          featureSamplingConfigPtr_(featureSamplingConfigPtr) {}

    uint32 AutoParallelRuleRefinementConfig::getNumThreads(const IFeatureMatrix& featureMatrix,
                                                           uint32 numOutputs) const {
        if (!lossConfigPtr_->isDecomposable() && !headConfigPtr_->isSingleOutput()) {
            return 1;
        } else if (featureMatrix.isSparse() && !featureSamplingConfigPtr_->isSamplingUsed()) {
            return 1;
        } else {
            return getNumAvailableThreads(0);
        }
    }

}
