#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a loss function that implements a multi-label
     * variant of the squared hinge loss that is applied label-wise.
     */
    class ILabelWiseSquaredHingeLossConfig {

        public:

            virtual ~ILabelWiseSquaredHingeLossConfig() { };

    };

    /**
     * Allows to configure a loss function that implements a multi-label variant of the squared hinge loss that is
     * applied label-wise.
     */
    class LabelWiseSquaredHingeLossConfig final : public ILabelWiseLossConfig, public ILabelWiseSquaredHingeLossConfig {

        public:

            std::unique_ptr<IStatisticsProviderFactory> configure() const override;

            std::unique_ptr<ILabelWiseLossFactory> configureLabelWise() const override;

    };

}
