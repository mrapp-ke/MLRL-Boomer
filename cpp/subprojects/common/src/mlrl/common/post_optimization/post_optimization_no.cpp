#include "mlrl/common/post_optimization/post_optimization_no.hpp"

std::unique_ptr<IPostOptimizationPhaseFactory> NoPostOptimizationPhaseConfig::createPostOptimizationPhaseFactory(
  const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const {
    return nullptr;
}
