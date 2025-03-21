/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a regularization term that affects the evaluation of
     * rules.
     */
    class IRegularizationConfig {
        public:

            virtual ~IRegularizationConfig() {}

            /**
             * Determines and returns the weight of the regularization term according to the specified configuration.
             *
             * @return The weight of the regularization term
             */
            virtual float32 getWeight() const = 0;
    };

}
