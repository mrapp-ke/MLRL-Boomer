#include "boosting/losses/loss_label_wise_squared_error.hpp"


namespace boosting {

    void LabelWiseSquaredErrorLoss::updateGradientAndHessian(bool trueLabel, float64 predictedScore, float64* gradient,
                                                             float64* hessian) const {
        float64 expectedScore = trueLabel ? 1 : -1;
        *gradient = (2 * predictedScore) - (2 * expectedScore);
        *hessian = 2;
    }

    float64 LabelWiseSquaredErrorLoss::evaluate(bool trueLabel, float64 predictedScore) const {
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 difference = (expectedScore - predictedScore);
        return difference * difference;
    }

}
