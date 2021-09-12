/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all (decomposable) loss functions that are applied label-wise and are suited for the use
     * of sparse data structures. To meet this requirement, the gradients and Hessians that are computed by the loss
     * function should be zero, if the prediction for a label is correct.
     */
    class ISparseLabelWiseLoss : virtual public ILabelWiseLoss {

        public:

            virtual ~ISparseLabelWiseLoss() { };

    };

}
