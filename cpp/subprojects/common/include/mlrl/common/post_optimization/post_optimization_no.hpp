/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/post_optimization/post_optimization.hpp"

#include <memory>

/**
 * Allows to configure a post optimization phase that does nothing.
 */
class NoPostOptimizationPhaseConfig final : public IPostOptimizationPhaseConfig {
    public:

        std::unique_ptr<IPostOptimizationPhaseFactory> createPostOptimizationPhaseFactory(
          const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const override;
};
