#include "mlrl/common/prediction/predictor_binary_no.hpp"

std::unique_ptr<IBinaryPredictorFactory> NoBinaryPredictorConfig::createPredictorFactory(
  const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
    return nullptr;
}

std::unique_ptr<ISparseBinaryPredictorFactory> NoBinaryPredictorConfig::createSparsePredictorFactory(
  const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return nullptr;
}

bool NoBinaryPredictorConfig::isLabelVectorSetNeeded() const {
    return false;
}
