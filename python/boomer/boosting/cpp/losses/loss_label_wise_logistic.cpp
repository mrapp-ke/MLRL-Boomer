#include "loss_label_wise_logistic.h"
#include <cmath>

using namespace boosting;


void LabelWiseLogisticLoss::updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                                     DenseVector<float64>::iterator hessian, bool trueLabel,
                                                     float64 predictedScore) const {
    float64 expectedScore = trueLabel ? 1 : -1;
    float64 exponential = std::exp(expectedScore * predictedScore);
    float64 expectedScorePow = expectedScore * expectedScore;
    float64 exponentialPow = exponential + 1;
    exponentialPow *= exponentialPow;
    *gradient = -expectedScore / (1 + exponential);
    *hessian = (expectedScorePow * exponential) / exponentialPow;
}
