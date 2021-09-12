#pragma once

#include "boosting/losses/loss_label_wise_sparse.hpp"


namespace boosting {

    /**
     * A multi-label variant of the squared hinge loss that is applied label-wise.
     */
    class LabelWiseSquaredHingeLoss final : public AbstractSparseLabelWiseLoss {

        public:

            LabelWiseSquaredHingeLoss();

    };

}
