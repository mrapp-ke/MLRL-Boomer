#include "loss_label_wise_logistic.h"
#include <cmath>


namespace boosting {

    /**
     * Calculates and returns the logistic function `1 / (1 + exp(-x))`, given a specific value `x`.
     *
     * This implementation exploits the identity `1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))` to increase numerical
     * stability (see, e.g., section "Numerically stable sigmoid function" in
     * https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
     *
     * @param x The value `x`
     * @return  The value that has been calculated
     */
    static inline float64 logisticFunction(float64 x) {
        if (x >= 0) {
            float64 exponential = std::exp(-x);  // Evaluates to 0 for large x, resulting in 1 ultimately
            return 1 / (1 + exponential);
        } else {
            float64 exponential = std::exp(x);  // Evaluates to 0 for large x, resulting in 0 ultimately
            return exponential / (1 + exponential);
        }
    }

    /**
     * Calculates and returns the function `1 / (1 + exp(-x))^2 = exp(x)^2 / (1 + exp(x))^2`, given a specific value
     * `x`.
     *
     * @param x The value `x`
     * @return  The value that has been calculated
     */
    static inline float64 squaredLogisticFunction(float64 x) {
        if (x >= 0) {
            float64 exponential = std::exp(-x);  // Evaluates to 0 for large x, resulting in 1 ultimately
            return 1 / ((exponential + 1) * (exponential + 1));
        } else {
            float64 exponential = std::exp(x);  // Evaluates to 0 for large x, resulting in 0 ultimately
            return (exponential * exponential) / ((exponential + 1) * (exponential + 1));
        }
    }

    void LabelWiseLogisticLoss::updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                                         DenseVector<float64>::iterator hessian, bool trueLabel,
                                                         float64 predictedScore) const {
        // The gradient computes as `-expectedScore / (1 + exp(expectedScore * predictedScore))`, or as
        // `1 / (1 + exp(-predictedScore)) - 1` if `trueLabel == true`, `1 / (1 + exp(-predictedScore))`, otherwise...
        float64 logistic = logisticFunction(predictedScore);
        *gradient = trueLabel ? logistic - 1.0 : logistic;

        // The Hessian computes as `exp(expectedScore * predictedScore) / (1 + exp(expectedScore * predictedScore))^2`,
        // or as `1 / (1 + exp(expectedScore * predictedScore)) - 1 / (1 + exp(expectedScore * predictedScore))^2`
        *hessian = logistic - squaredLogisticFunction(predictedScore);
    }

}
