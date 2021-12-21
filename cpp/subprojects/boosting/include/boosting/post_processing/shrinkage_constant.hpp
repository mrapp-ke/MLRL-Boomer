/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/post_processing/post_processor.hpp"


namespace boosting {

    /**
     * Allows to create instances of the type `IPostProcessor` that post-process the predictions of rules by shrinking
     * their weights by a constant shrinkage parameter.
     */
    class ConstantShrinkageFactory final : public IPostProcessorFactory {

        private:

            float64 shrinkage_;

        public:

            /**
             * @param shrinkage The shrinkage parameter. Must be in (0, 1)
             */
            ConstantShrinkageFactory(float64 shrinkage);

            std::unique_ptr<IPostProcessor> create() const override;

    };

}
