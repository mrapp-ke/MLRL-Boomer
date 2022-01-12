/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss_example_wise.hpp"


namespace boosting {

    /**
     * Allows to configure a loss function that implements a multi-label variant of the logistic loss that is applied
     * example-wise.
     */
    class ExampleWiseLogisticLossConfig final : public ILossConfig {

    };

    /**
     * Allows to create instances of the type `IExampleWiseLoss` that implement a multi-label variant of the logistic
     * loss that is applied example-wise.
     */
    class ExampleWiseLogisticLossFactory final : public IExampleWiseLossFactory {

        public:

            std::unique_ptr<IExampleWiseLoss> createExampleWiseLoss() const override;

    };

}
