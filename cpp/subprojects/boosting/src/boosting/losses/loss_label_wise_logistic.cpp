#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/math/math.hpp"
#include "loss_label_wise_common.hpp"


namespace boosting {

    /**
     * Calculates and returns the function `1 / (1 + exp(-x))^2 = exp(x)^2 / (1 + exp(x))^2`, given a specific value
     * `x`.
     *
     * @param x The value `x`
     * @return  The value that has been calculated
     */
    static inline constexpr float64 squaredLogisticFunction(float64 x) {
        if (x >= 0) {
            float64 exponential = std::exp(-x);  // Evaluates to 0 for large x, resulting in 1 ultimately
            return 1 / ((exponential + 1) * (exponential + 1));
        } else {
            float64 exponential = std::exp(x);  // Evaluates to 0 for large x, resulting in 0 ultimately
            return (exponential * exponential) / ((exponential + 1) * (exponential + 1));
        }
    }

    /**
     * Calculates and returns the function `log(1 + exp(x)) = log(exp(0) + exp(x))`, given a specific value `x`.
     *
     * This function exploits the identity `log(exp(0) + exp(x)) = b + log(exp(0 - b) + exp(x - b))`, where
     * `b = max(0, x)`, to increase numerical stability (see, e.g., section "Log-sum-exp for computing the
     * log-distribution" in https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
     *
     * @param x The value `x`
     * @return  The value that has been calculated
     */
    static inline constexpr float64 logSumExp(float64 x) {
        if (x > 0) {
            return x + std::log(std::exp(0 - x) + 1);
        } else {
            return std::log(1 + std::exp(x));
        }
    }

    static inline void updateGradientAndHessian(bool trueLabel, float64 predictedScore, float64* gradient,
                                                float64* hessian) {
        // The gradient computes as `-expectedScore / (1 + exp(expectedScore * predictedScore))`, or as
        // `1 / (1 + exp(-predictedScore)) - 1` if `trueLabel == true`, `1 / (1 + exp(-predictedScore))`, otherwise...
        float64 logistic = logisticFunction(predictedScore);
        *gradient = trueLabel ? logistic - 1.0 : logistic;

        // The Hessian computes as `exp(expectedScore * predictedScore) / (1 + exp(expectedScore * predictedScore))^2`,
        // or as `1 / (1 + exp(expectedScore * predictedScore)) - 1 / (1 + exp(expectedScore * predictedScore))^2`
        *hessian = logistic - squaredLogisticFunction(predictedScore);
    }

    static inline float64 evaluatePrediction(bool trueLabel, float64 predictedScore) {
       // The logistic loss calculates as `log(1 + exp(-expectedScore * predictedScore))`...
        float64 x = trueLabel ? -predictedScore : predictedScore;
        return logSumExp(x);
    }

    /**
     * Allows to create instances of the type `ILabelWiseLoss` that implement a multi-label variant of the logistic loss
     * that is applied label-wise.
     */
    class LabelWiseLogisticLossFactory final : public ILabelWiseLossFactory {

        public:

            std::unique_ptr<ILabelWiseLoss> createLabelWiseLoss() const override {
                return std::make_unique<LabelWiseLoss>(&updateGradientAndHessian, &evaluatePrediction);
            }

    };

    /**
     * Allows to transform the score that is predicted for an individual label into a probability by applying the
     * logistic sigmoid function.
     */
    class LogisticFunction final : public IProbabilityFunction {

        public:

            float64 transform(float64 predictedScore) const override {
                return logisticFunction(predictedScore);
            }

    };

    /**
     * Allows to create instances of the type `IProbabilityFunction` that transform the score that is predicted for an
     * individual label into a probability by applying the logistic sigmoid function.
     */
    class LogisticFunctionFactory final : public IProbabilityFunctionFactory {

        public:

            std::unique_ptr<IProbabilityFunction> create() const override {
                return std::make_unique<LogisticFunction>();
            }

    };

    LabelWiseLogisticLossConfig::LabelWiseLogisticLossConfig(const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : headConfigPtr_(headConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> LabelWiseLogisticLossConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix, const Blas& blas,
            const Lapack& lapack) const {
        return headConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, *this);
    }

    std::unique_ptr<IProbabilityFunctionFactory> LabelWiseLogisticLossConfig::createProbabilityFunctionFactory() const {
        return std::make_unique<LogisticFunctionFactory>();
    }

    float64 LabelWiseLogisticLossConfig::getDefaultPrediction() const {
        return 0;
    }

    std::unique_ptr<ILabelWiseLossFactory> LabelWiseLogisticLossConfig::createLabelWiseLossFactory() const {
        return std::make_unique<LabelWiseLogisticLossFactory>();
    }

}
