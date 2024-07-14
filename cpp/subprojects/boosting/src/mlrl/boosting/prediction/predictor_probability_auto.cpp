#include "mlrl/boosting/prediction/predictor_probability_auto.hpp"

#include "mlrl/boosting/prediction/predictor_probability_marginalized.hpp"
#include "mlrl/boosting/prediction/predictor_probability_output_wise.hpp"

namespace boosting {

    AutomaticProbabilityPredictorConfig::AutomaticProbabilityPredictorConfig(
      GetterFunction<ILossConfig> lossConfigGetter, GetterFunction<IMultiThreadingConfig> multiThreadingConfigGetter)
        : lossConfigGetter_(lossConfigGetter), multiThreadingConfigGetter_(multiThreadingConfigGetter) {}

    std::unique_ptr<IProbabilityPredictorFactory> AutomaticProbabilityPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        if (lossConfigGetter_().isDecomposable()) {
            return OutputWiseProbabilityPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .createPredictorFactory(featureMatrix, numOutputs);
        } else {
            return MarginalizedProbabilityPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .createPredictorFactory(featureMatrix, numOutputs);
        }
    }

    bool AutomaticProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        if (lossConfigGetter_().isDecomposable()) {
            return OutputWiseProbabilityPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .isLabelVectorSetNeeded();
        } else {
            return MarginalizedProbabilityPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .isLabelVectorSetNeeded();
        }
    }

}
