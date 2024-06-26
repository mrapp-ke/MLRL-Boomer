#include "mlrl/boosting/prediction/predictor_binary_auto.hpp"

#include "mlrl/boosting/prediction/predictor_binary_example_wise.hpp"
#include "mlrl/boosting/prediction/predictor_binary_output_wise.hpp"

namespace boosting {

    AutomaticBinaryPredictorConfig::AutomaticBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IBinaryPredictorFactory> AutomaticBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        if (lossConfigPtr_->isDecomposable()) {
            return OutputWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
              .createPredictorFactory(featureMatrix, numOutputs);
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
              .createPredictorFactory(featureMatrix, numOutputs);
        }
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> AutomaticBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (lossConfigPtr_->isDecomposable()) {
            return OutputWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
              .createSparsePredictorFactory(featureMatrix, numLabels);
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
              .createSparsePredictorFactory(featureMatrix, numLabels);
        }
    }

    bool AutomaticBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        if (lossConfigPtr_->isDecomposable()) {
            return OutputWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_).isLabelVectorSetNeeded();
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_).isLabelVectorSetNeeded();
        }
    }

}
