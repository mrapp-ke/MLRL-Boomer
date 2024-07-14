#include "mlrl/boosting/prediction/predictor_probability_auto.hpp"

#include "mlrl/boosting/prediction/predictor_probability_marginalized.hpp"
#include "mlrl/boosting/prediction/predictor_probability_output_wise.hpp"

namespace boosting {

    AutomaticProbabilityPredictorConfig::AutomaticProbabilityPredictorConfig(
      ReadableProperty<IClassificationLossConfig> lossConfigGetter,
      ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter)
        : lossConfig_(lossConfigGetter), multiThreadingConfig_(multiThreadingConfigGetter) {}

    std::unique_ptr<IProbabilityPredictorFactory> AutomaticProbabilityPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        if (lossConfig_.get().isDecomposable()) {
            return OutputWiseProbabilityPredictorConfig(lossConfig_, multiThreadingConfig_)
              .createPredictorFactory(featureMatrix, numOutputs);
        } else {
            return MarginalizedProbabilityPredictorConfig(lossConfig_, multiThreadingConfig_)
              .createPredictorFactory(featureMatrix, numOutputs);
        }
    }

    bool AutomaticProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        if (lossConfig_.get().isDecomposable()) {
            return OutputWiseProbabilityPredictorConfig(lossConfig_, multiThreadingConfig_).isLabelVectorSetNeeded();
        } else {
            return MarginalizedProbabilityPredictorConfig(lossConfig_, multiThreadingConfig_).isLabelVectorSetNeeded();
        }
    }

}
