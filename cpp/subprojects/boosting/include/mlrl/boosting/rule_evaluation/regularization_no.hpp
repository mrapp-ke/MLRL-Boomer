/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/regularization.hpp"

namespace boosting {

    /**
     * Allows to configure a regularization term that does not affect the evaluation of rules.
     */
    class NoRegularizationConfig final : public IRegularizationConfig {
        public:

            float32 getWeight() const override;
    };

}
