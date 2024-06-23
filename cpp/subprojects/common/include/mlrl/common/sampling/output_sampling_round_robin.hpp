/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/output_sampling.hpp"

#include <memory>

/**
 * Allows to configure a method for sampling outputs in a round-robin fashion.
 */
class RoundRobinOutputSamplingConfig final : public IOutputSamplingConfig {
    public:

        std::unique_ptr<IOutputSamplingFactory> createOutputSamplingFactory(
          const IOutputMatrix& outputMatrix) const override;
};
