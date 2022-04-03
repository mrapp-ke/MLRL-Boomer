#include "boosting/output/predictor_probability_auto.hpp"
#include "boosting/output/predictor_probability_label_wise.hpp"
#include "boosting/output/predictor_probability_marginalized.hpp"
#include "boosting/losses/loss_example_wise.hpp"


namespace boosting {

    static inline bool isExampleWisePredictorPreferred(const ILossConfig* lossConfig) {
        return dynamic_cast<const IExampleWiseLossConfig*>(lossConfig) != nullptr;
    }

    AutomaticProbabilityPredictorConfig::AutomaticProbabilityPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<IProbabilityPredictorFactory> AutomaticProbabilityPredictorConfig::createProbabilityPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (isExampleWisePredictorPreferred(lossConfigPtr_.get())) {
            return MarginalizedProbabilityPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createProbabilityPredictorFactory(featureMatrix, numLabels);
        } else {
            return LabelWiseProbabilityPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createProbabilityPredictorFactory(featureMatrix, numLabels);
        }
    }

}
