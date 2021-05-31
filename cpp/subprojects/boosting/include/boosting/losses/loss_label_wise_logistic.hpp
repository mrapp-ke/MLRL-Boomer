/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * A multi-label variant of the logistic loss that is applied label-wise.
     */
    class LabelWiseLogisticLoss final : public AbstractLabelWiseLoss {

        protected:

            void updateGradientAndHessian(bool trueLabel, float64 predictedScore, float64* gradient,
                                          float64* hessian) const override;

            float64 evaluate(bool trueLabel, float64 predictedScore) const override;

    };

}
