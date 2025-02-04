/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/multi_threading/multi_threading.hpp"

/**
 * Allows to configure the multi-threading behavior of a parallelize algorithm that should not use any multi-threading.
 */
class NoMultiThreadingConfig final : public IMultiThreadingConfig {
    public:

        MultiThreadingSettings getSettings(const IFeatureMatrix& featureMatrix, uint32 numOutputs) const override;
};
