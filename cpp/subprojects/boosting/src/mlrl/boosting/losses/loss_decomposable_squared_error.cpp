#include "mlrl/boosting/losses/loss_decomposable_squared_error.hpp"

#include "loss_decomposable_common.hpp"

namespace boosting {

    static inline void updateGradientAndHessian(bool trueLabel, float64 predictedScore, float64& gradient,
                                                float64& hessian) {
        float64 expectedScore = trueLabel ? 1 : -1;
        gradient = (predictedScore - expectedScore);
        hessian = 1;
    }

    static inline float64 evaluatePrediction(bool trueLabel, float64 predictedScore) {
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 difference = (expectedScore - predictedScore);
        return difference * difference;
    }

    /**
     * Allows to create instances of the type `IDecomposableLoss` that implement a multivariate variant of the squared
     * error loss that is decomposable.
     */
    class DecomposableSquaredErrorLossFactory final : public IDecomposableLossFactory {
        public:

            std::unique_ptr<IDecomposableLoss> createDecomposableLoss() const override {
                return std::make_unique<DecomposableLoss>(&updateGradientAndHessian, &evaluatePrediction);
            }
    };

    DecomposableSquaredErrorLossConfig::DecomposableSquaredErrorLossConfig(GetterFunction<IHeadConfig> headConfigGetter)
        : headConfigGetter_(headConfigGetter) {}

    std::unique_ptr<IStatisticsProviderFactory> DecomposableSquaredErrorLossConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack, bool preferSparseStatistics) const {
        return headConfigGetter_().createStatisticsProviderFactory(featureMatrix, labelMatrix, *this);
    }

    std::unique_ptr<IMarginalProbabilityFunctionFactory>
      DecomposableSquaredErrorLossConfig::createMarginalProbabilityFunctionFactory() const {
        return nullptr;
    }

    std::unique_ptr<IJointProbabilityFunctionFactory>
      DecomposableSquaredErrorLossConfig::createJointProbabilityFunctionFactory() const {
        return nullptr;
    }

    float64 DecomposableSquaredErrorLossConfig::getDefaultPrediction() const {
        return 0;
    }

    std::unique_ptr<IDecomposableLossFactory> DecomposableSquaredErrorLossConfig::createDecomposableLossFactory()
      const {
        return std::make_unique<DecomposableSquaredErrorLossFactory>();
    }

}
