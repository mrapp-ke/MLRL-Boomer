#include "mlrl/boosting/prediction/predictor_binary_auto.hpp"

#include "mlrl/boosting/prediction/predictor_binary_example_wise.hpp"
#include "mlrl/boosting/prediction/predictor_binary_output_wise.hpp"

namespace boosting {

    AutomaticBinaryPredictorConfig::AutomaticBinaryPredictorConfig(
      ReadableProperty<IClassificationLossConfig> lossConfigGetter,
      ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter)
        : lossConfig_(lossConfigGetter), multiThreadingConfig_(multiThreadingConfigGetter) {}

    std::unique_ptr<IBinaryPredictorFactory> AutomaticBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        if (lossConfig_.get().isDecomposable()) {
            return OutputWiseBinaryPredictorConfig(lossConfig_, multiThreadingConfig_)
              .createPredictorFactory(featureMatrix, numOutputs);
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfig_, multiThreadingConfig_)
              .createPredictorFactory(featureMatrix, numOutputs);
        }
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> AutomaticBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (lossConfig_.get().isDecomposable()) {
            return OutputWiseBinaryPredictorConfig(lossConfig_, multiThreadingConfig_)
              .createSparsePredictorFactory(featureMatrix, numLabels);
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfig_, multiThreadingConfig_)
              .createSparsePredictorFactory(featureMatrix, numLabels);
        }
    }

    bool AutomaticBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        if (lossConfig_.get().isDecomposable()) {
            return OutputWiseBinaryPredictorConfig(lossConfig_, multiThreadingConfig_).isLabelVectorSetNeeded();
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfig_, multiThreadingConfig_).isLabelVectorSetNeeded();
        }
    }

}
