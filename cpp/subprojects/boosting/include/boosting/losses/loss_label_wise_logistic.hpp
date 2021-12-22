/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * An implementation of the type `ILabelWiseLoss` that implements a multi-label variant of the logistic loss that is
     * applied label-wise.
     */
    class LabelWiseLogisticLoss final : public AbstractLabelWiseLoss {

        public:

            LabelWiseLogisticLoss();

    };

    /**
     * Allows to create instances of the type `ILabelWiseLoss` that implement a multi-label variant of the logistic loss
     * that is applied label-wise.
     */
    class LabelWiseLogisticLossFactory final : public ILabelWiseLossFactory {

        public:

            std::unique_ptr<ILabelWiseLoss> create() const override;

    };

}
