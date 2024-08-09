#include "mlrl/common/prediction/predictor_score_no.hpp"

std::unique_ptr<IScorePredictorFactory> NoScorePredictorConfig::createPredictorFactory(
  const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
    return nullptr;
}

bool NoScorePredictorConfig::isLabelVectorSetNeeded() const {
    return false;
}
