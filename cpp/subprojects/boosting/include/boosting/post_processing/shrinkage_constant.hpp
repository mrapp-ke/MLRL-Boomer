/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/post_processing/post_processor.hpp"


namespace boosting {

    /**
     * Allows to configure a post-processor that shrinks the weights of rules by a constant "shrinkage" parameter.
     */
    class ConstantShrinkageConfig final : public IPostProcessorConfig {

        private:

            float64 shrinkage_;

        public:

            /**
             * Returns the value of the "shrinkage" parameter.
             *
             * @return The value of the "shrinkage" parameter
             */
            float64 getShrinkage() const;

            /**
             * Sets the value of the "shrinkage" parameter.
             *
             * @param shrinkage The value of the "shrinkage" parameter. Must be in (0, 1)
             * @return          A reference to an object of type `ConstantShrinkageConfig` that allows further
             *                  configuration of the post-processor
             */
            ConstantShrinkageConfig& setShrinkage(float64 shrinkage);

    };

    /**
     * Allows to create instances of the type `IPostProcessor` that post-process the predictions of rules by shrinking
     * their weights by a constant "shrinkage" parameter.
     */
    class ConstantShrinkageFactory final : public IPostProcessorFactory {

        private:

            float64 shrinkage_;

        public:

            /**
             * @param shrinkage The value of the "shrinkage" parameter. Must be in (0, 1)
             */
            ConstantShrinkageFactory(float64 shrinkage);

            std::unique_ptr<IPostProcessor> create() const override;

    };

}
