#include "mlrl/common/prediction/predictor_probability_no.hpp"

std::unique_ptr<IProbabilityPredictorFactory> NoProbabilityPredictorConfig::createPredictorFactory(
  const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
    return nullptr;
}

bool NoProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
    return false;
}
