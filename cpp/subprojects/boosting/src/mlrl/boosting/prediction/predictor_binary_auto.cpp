#include "mlrl/boosting/prediction/predictor_binary_auto.hpp"

#include "mlrl/boosting/prediction/predictor_binary_example_wise.hpp"
#include "mlrl/boosting/prediction/predictor_binary_output_wise.hpp"

namespace boosting {

    AutomaticBinaryPredictorConfig::AutomaticBinaryPredictorConfig(
      GetterFunction<ILossConfig> lossConfigGetter, GetterFunction<IMultiThreadingConfig> multiThreadingConfigGetter)
        : lossConfigGetter_(lossConfigGetter), multiThreadingConfigGetter_(multiThreadingConfigGetter) {}

    std::unique_ptr<IBinaryPredictorFactory> AutomaticBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        if (lossConfigGetter_().isDecomposable()) {
            return OutputWiseBinaryPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .createPredictorFactory(featureMatrix, numOutputs);
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .createPredictorFactory(featureMatrix, numOutputs);
        }
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> AutomaticBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (lossConfigGetter_().isDecomposable()) {
            return OutputWiseBinaryPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .createSparsePredictorFactory(featureMatrix, numLabels);
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .createSparsePredictorFactory(featureMatrix, numLabels);
        }
    }

    bool AutomaticBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        if (lossConfigGetter_().isDecomposable()) {
            return OutputWiseBinaryPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .isLabelVectorSetNeeded();
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfigGetter_, multiThreadingConfigGetter_)
              .isLabelVectorSetNeeded();
        }
    }

}
