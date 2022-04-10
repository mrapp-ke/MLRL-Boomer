#include "boosting/output/predictor_probability_auto.hpp"
#include "boosting/output/predictor_probability_label_wise.hpp"
#include "boosting/output/predictor_probability_marginalized.hpp"


namespace boosting {

    AutomaticProbabilityPredictorConfig::AutomaticProbabilityPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<IProbabilityPredictorFactory> AutomaticProbabilityPredictorConfig::createProbabilityPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseProbabilityPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createProbabilityPredictorFactory(featureMatrix, numLabels);
        } else {
            return MarginalizedProbabilityPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createProbabilityPredictorFactory(featureMatrix, numLabels);
        }
    }

    bool AutomaticProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseProbabilityPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .isLabelVectorSetNeeded();
        } else {
            return MarginalizedProbabilityPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .isLabelVectorSetNeeded();
        }
    }

}
