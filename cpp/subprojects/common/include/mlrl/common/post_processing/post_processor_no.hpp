/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/post_processing/post_processor.hpp"

#include <memory>

/**
 * Allows to configure a post-processor that does not perform any post-processing, but retains the original predictions
 * of rules.
 */
class NoPostProcessorConfig final : public IPostProcessorConfig {
    public:

        std::unique_ptr<IPostProcessorFactory> createPostProcessorFactory() const override;
};
