#include "mlrl/boosting/losses/loss_decomposable_logistic.hpp"

#include "loss_decomposable_common.hpp"
#include "mlrl/boosting/prediction/probability_function_chain_rule.hpp"
#include "mlrl/boosting/prediction/probability_function_logistic.hpp"
#include "mlrl/boosting/util/math.hpp"

namespace boosting {

    /**
     * Calculates and returns the function `1 / (1 + exp(-x))^2 = exp(x)^2 / (1 + exp(x))^2`, given a specific value
     * `x`.
     *
     * This implementation exploits the identity `1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))` to increase numerical
     * stability (see, e.g., section "Numerically stable sigmoid function" in
     * https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
     *
     * @tparam T    The type of the value `x`
     * @param x     The value `x`
     * @return      The value that has been calculated
     */
    template<typename T>
    static inline constexpr T squaredLogisticFunction(T x) {
        if (x < 0) {
            T exponential = std::exp(x);  // Evaluates to 0 for large x, resulting in 0 ultimately
            return (exponential * exponential) / ((exponential + 1) * (exponential + 1));
        } else {
            T exponential = std::exp(-x);  // Evaluates to 0 for large x, resulting in 1 ultimately
            return 1 / ((exponential + 1) * (exponential + 1));
        }
    }

    /**
     * Calculates and returns the function `log(1 + exp(x)) = log(exp(0) + exp(x))`, given a specific value `x`.
     *
     * This function exploits the identity `log(exp(0) + exp(x)) = b + log(exp(0 - b) + exp(x - b))`, where
     * `b = max(0, x)`, to increase numerical stability (see, e.g., section "Log-sum-exp for computing the
     * log-distribution" in https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
     *
     * @tparam T    The type of the value `x`
     * @param x     The value `x`
     * @return      The value that has been calculated
     */
    template<typename T>
    static inline constexpr T logSumExp(T x) {
        if (x > 0) {
            return x + std::log(std::exp(0 - x) + 1);
        } else {
            return std::log(1 + std::exp(x));
        }
    }

    template<typename StatisticType>
    static inline void updateGradientAndHessian(bool trueLabel, StatisticType predictedScore, StatisticType& gradient,
                                                StatisticType& hessian) {
        // The gradient computes as `-expectedScore / (1 + exp(expectedScore * predictedScore))`, or as
        // `1 / (1 + exp(-predictedScore)) - 1` if `trueLabel == true`, `1 / (1 + exp(-predictedScore))`, otherwise...
        StatisticType logistic = util::logisticFunction(predictedScore);
        gradient = trueLabel ? logistic - 1.0 : logistic;

        // The Hessian computes as `exp(expectedScore * predictedScore) / (1 + exp(expectedScore * predictedScore))^2`,
        // or as `1 / (1 + exp(expectedScore * predictedScore)) - 1 / (1 + exp(expectedScore * predictedScore))^2`
        hessian = logistic - squaredLogisticFunction(predictedScore);
    }

    template<typename ScoreType>
    static inline ScoreType evaluatePrediction(bool trueLabel, ScoreType predictedScore) {
        // The logistic loss calculates as `log(1 + exp(-expectedScore * predictedScore))`...
        ScoreType x = trueLabel ? -predictedScore : predictedScore;
        return logSumExp(x);
    }

    /**
     * Allows to create instances of the type `IDecomposableClassificationLoss` that implement a multivariate variant of
     * the logistic loss that is decomposable.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class DecomposableLogisticLossFactory final : public IDecomposableClassificationLossFactory<StatisticType> {
        public:

            std::unique_ptr<IDecomposableClassificationLoss<StatisticType>> createDecomposableClassificationLoss()
              const override {
                return std::make_unique<DecomposableClassificationLoss<StatisticType>>(
                  &updateGradientAndHessian<StatisticType>, &evaluatePrediction<StatisticType>);
            }

            std::unique_ptr<IDistanceMeasure<StatisticType>> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const override {
                return this->createDecomposableClassificationLoss();
            }

            std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> createClassificationEvaluationMeasure()
              const override {
                return this->createDecomposableClassificationLoss();
            }
    };

    template<typename StatisticType>
    class DecomposableLogisticLossPreset final : public IDecomposableClassificationLossConfig::IPreset<StatisticType> {
        public:

            std::unique_ptr<IDecomposableClassificationLossFactory<StatisticType>>
              createDecomposableClassificationLossFactory() const override {
                return std::make_unique<DecomposableLogisticLossFactory<StatisticType>>();
            }
    };

    DecomposableLogisticLossConfig::DecomposableLogisticLossConfig(
      ReadableProperty<IStatisticTypeConfig> statisticTypeConfig)
        : statisticTypeConfig_(statisticTypeConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      DecomposableLogisticLossConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return statisticTypeConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix,
                                                                                        *this);
    }

    std::unique_ptr<IMarginalProbabilityFunctionFactory>
      DecomposableLogisticLossConfig::createMarginalProbabilityFunctionFactory() const {
        return std::make_unique<LogisticFunctionFactory>();
    }

    std::unique_ptr<IJointProbabilityFunctionFactory>
      DecomposableLogisticLossConfig::createJointProbabilityFunctionFactory() const {
        return std::make_unique<ChainRuleFactory>(this->createMarginalProbabilityFunctionFactory());
    }

    float64 DecomposableLogisticLossConfig::getDefaultPrediction() const {
        return 0;
    }

    std::unique_ptr<IDecomposableClassificationLossConfig::IPreset<float32>>
      DecomposableLogisticLossConfig::createDecomposable32BitClassificationPreset() const {
        return std::make_unique<DecomposableLogisticLossPreset<float32>>();
    }

    std::unique_ptr<IDecomposableClassificationLossConfig::IPreset<float64>>
      DecomposableLogisticLossConfig::createDecomposable64BitClassificationPreset() const {
        return std::make_unique<DecomposableLogisticLossPreset<float64>>();
    }

}
