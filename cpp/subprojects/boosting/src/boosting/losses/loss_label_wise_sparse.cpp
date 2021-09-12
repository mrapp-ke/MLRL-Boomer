#include "boosting/losses/loss_label_wise_sparse.hpp"


namespace boosting {

    AbstractSparseLabelWiseLoss::AbstractSparseLabelWiseLoss(UpdateFunction updateFunction,
                                                             EvaluateFunction evaluateFunction)
        : AbstractLabelWiseLoss(updateFunction, evaluateFunction) {

    }

}
