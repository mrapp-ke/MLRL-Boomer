/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss_example_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a loss function that implements a multi-label
     * variant of the logistic loss that is applied example-wise.
     */
    class IExampleWiseLogisticLossConfig {

        public:

            virtual ~IExampleWiseLogisticLossConfig() { };

    };

    /**
     * Allows to configure a loss function that implements a multi-label variant of the logistic loss that is applied
     * example-wise.
     */
    class ExampleWiseLogisticLossConfig final : public IExampleWiseLossConfig, public IExampleWiseLogisticLossConfig {

        public:

            std::unique_ptr<IStatisticsProviderFactory> configure() const override;

            std::unique_ptr<IExampleWiseLossFactory> configureExampleWise() const override;

    };

}
