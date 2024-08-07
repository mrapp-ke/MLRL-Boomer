/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/instance_sampling.hpp"

#include <memory>

/**
 * Allows to configure a method for sampling training examples that does not perform any sampling, but assigns equal
 * weights to all examples.
 */
class NoInstanceSamplingConfig final : public IClassificationInstanceSamplingConfig,
                                       public IRegressionInstanceSamplingConfig {
    public:

        std::unique_ptr<IClassificationInstanceSamplingFactory> createClassificationInstanceSamplingFactory()
          const override;

        std::unique_ptr<IRegressionInstanceSamplingFactory> createRegressionInstanceSamplingFactory() const override;
};
