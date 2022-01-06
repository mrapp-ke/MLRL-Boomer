/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * Allows to create instances of the type `ILabelWiseLoss` that implement a multi-label variant of the logistic loss
     * that is applied label-wise.
     */
    class LabelWiseLogisticLossFactory final : virtual public ILabelWiseLossFactory {

        public:

            std::unique_ptr<ILabelWiseLoss> createLabelWiseLoss() const override;

    };

}
