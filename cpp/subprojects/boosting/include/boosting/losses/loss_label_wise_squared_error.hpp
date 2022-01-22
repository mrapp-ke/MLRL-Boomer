/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a loss function that implements a multi-label
     * variant of the squared error loss that is applied label-wise.
     */
    class ILabelWiseSquaredErrorLossConfig {

        public:

            virtual ~ILabelWiseSquaredErrorLossConfig() { };

    };

    /**
     * Allows to configure a loss function that implements a multi-label variant of the squared error loss that is
     * applied label-wise.
     */
    class LabelWiseSquaredErrorLossConfig final : public ILossConfig, public ILabelWiseSquaredErrorLossConfig {

        public:

            std::unique_ptr<IStatisticsProviderFactory> configure() const override;

    };

}
