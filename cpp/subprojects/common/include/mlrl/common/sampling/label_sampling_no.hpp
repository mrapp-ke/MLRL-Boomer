/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/label_sampling.hpp"

#include <memory>

/**
 * Allows to configure a method for sampling labels that does not perform any sampling, but includes all labels.
 */
class NoLabelSamplingConfig final : public ILabelSamplingConfig {
    public:

        std::unique_ptr<ILabelSamplingFactory> createLabelSamplingFactory(
          const ILabelMatrix& labelMatrix) const override;
};