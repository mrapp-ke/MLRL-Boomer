/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/post_processing/post_processor.hpp"


/**
 * Allows to create instances of the type `IPostProcessor` that do not perform any post-processing, but retain the
 * original predictions of rules.
 */
class NoPostProcessorFactory final : public IPostProcessorFactory {

    public:

        std::unique_ptr<IPostProcessor> create() const override;

};
