#include "loss_label_wise_squared_hinge.h"


namespace boosting {

    void LabelWiseSquaredErrorLoss::updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                                             DenseVector<float64>::iterator hessian, bool trueLabel,
                                                             float64 predictedScore) const {
        if (trueLabel) {
            if (predictedScore < 1) {
                *gradient = 2 * (predictedScore - 1);
                *hessian = 2;
            } else {
                *gradient = 0;
                *hessian = 0;
            }
        } else {
            if (predictedScore > 0) {
                *gradient = 2 * predictedScore;
                *hessian = 2;
            } else {
                *gradient = 0;
                *hessian = 0;
            }
        }
    }

}
