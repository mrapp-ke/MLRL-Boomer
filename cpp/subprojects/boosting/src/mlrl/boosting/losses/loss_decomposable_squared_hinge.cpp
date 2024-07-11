#include "mlrl/boosting/losses/loss_decomposable_squared_hinge.hpp"

#include "loss_decomposable_sparse_common.hpp"

namespace boosting {

    static inline void updateGradientAndHessian(bool trueLabel, float64 predictedScore, float64& gradient,
                                                float64& hessian) {
        if (trueLabel) {
            if (predictedScore < 1) {
                gradient = (predictedScore - 1);
            } else {
                gradient = 0;
            }
        } else {
            if (predictedScore > 0) {
                gradient = predictedScore;
            } else {
                gradient = 0;
            }
        }

        hessian = 1;
    }

    static inline float64 evaluatePrediction(bool trueLabel, float64 predictedScore) {
        if (trueLabel) {
            if (predictedScore < 1) {
                return (1 - predictedScore) * (1 - predictedScore);
            } else {
                return 0;
            }
        } else {
            if (predictedScore > 0) {
                return predictedScore * predictedScore;
            } else {
                return 0;
            }
        }
    }

    /**
     * Allows to create instances of the type `IDecomposableLoss` that implement a multivariate variant of the squared
     * hinge loss that is decomposable.
     */
    class DecomposableSquaredHingeLossFactory final : public ISparseDecomposableLossFactory {
        public:

            std::unique_ptr<ISparseDecomposableLoss> createSparseDecomposableLoss() const override {
                return std::make_unique<SparseDecomposableLoss>(&updateGradientAndHessian, &evaluatePrediction);
            }
    };

    DecomposableSquaredHingeLossConfig::DecomposableSquaredHingeLossConfig(
      ReadableProperty<IHeadConfig> headConfigGetter)
        : headConfig_(headConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      DecomposableSquaredHingeLossConfig::createStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                          const IRowWiseLabelMatrix& labelMatrix,
                                                                          const Blas& blas, const Lapack& lapack,
                                                                          bool preferSparseStatistics) const {
        if (preferSparseStatistics) {
            return headConfig_.get().createStatisticsProviderFactory(featureMatrix, labelMatrix, *this);
        } else {
            return headConfig_.get().createStatisticsProviderFactory(
              featureMatrix, labelMatrix, static_cast<const IDecomposableLossConfig&>(*this));
        }
    }

    std::unique_ptr<IMarginalProbabilityFunctionFactory>
      DecomposableSquaredHingeLossConfig::createMarginalProbabilityFunctionFactory() const {
        return nullptr;
    }

    std::unique_ptr<IJointProbabilityFunctionFactory>
      DecomposableSquaredHingeLossConfig::createJointProbabilityFunctionFactory() const {
        return nullptr;
    }

    float64 DecomposableSquaredHingeLossConfig::getDefaultPrediction() const {
        return 0.5;
    }

    std::unique_ptr<ISparseDecomposableLossFactory>
      DecomposableSquaredHingeLossConfig::createSparseDecomposableLossFactory() const {
        return std::make_unique<DecomposableSquaredHingeLossFactory>();
    }

}
