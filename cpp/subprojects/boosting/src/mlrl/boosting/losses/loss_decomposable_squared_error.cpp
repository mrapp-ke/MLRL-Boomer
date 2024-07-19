#include "mlrl/boosting/losses/loss_decomposable_squared_error.hpp"

#include "loss_decomposable_common.hpp"

namespace boosting {

    static inline void updateGradientAndHessianRegression(float32 expectedScore, float64 predictedScore,
                                                          float64& gradient, float64& hessian) {
        gradient = (predictedScore - (float64) expectedScore);
        hessian = 1;
    }

    static inline void updateGradientAndHessianClassification(bool trueLabel, float64 predictedScore, float64& gradient,
                                                              float64& hessian) {
        updateGradientAndHessianRegression(trueLabel ? 1 : -1, predictedScore, gradient, hessian);
    }

    static inline float64 evaluatePredictionRegression(float32 expectedScore, float64 predictedScore) {
        float64 difference = ((float64) expectedScore - predictedScore);
        return difference * difference;
    }

    static inline float64 evaluatePredictionClassification(bool trueLabel, float64 predictedScore) {
        return evaluatePredictionRegression(trueLabel ? 1 : -1, predictedScore);
    }

    /**
     * Allows to create instances of the type `IDecomposableClassificationLoss` that implement a multivariate variant of
     * the squared error loss that is decomposable.
     */
    class DecomposableSquaredErrorLossFactory final : public IDecomposableClassificationLossFactory,
                                                      public IDecomposableRegressionLossFactory {
        public:

            std::unique_ptr<IDecomposableClassificationLoss> createDecomposableClassificationLoss() const override {
                return std::make_unique<DecomposableClassificationLoss>(&updateGradientAndHessianClassification,
                                                                        &evaluatePredictionClassification);
            }

            std::unique_ptr<IDecomposableRegressionLoss> createDecomposableRegressionLoss() const override {
                return std::make_unique<DecomposableRegressionLoss>(&updateGradientAndHessianRegression,
                                                                    &evaluatePredictionRegression);
            }

            std::unique_ptr<IDistanceMeasure> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const {
                return this->createDecomposableClassificationLoss();
            }

            std::unique_ptr<IClassificationEvaluationMeasure> createClassificationEvaluationMeasure() const {
                return this->createDecomposableClassificationLoss();
            }

            std::unique_ptr<IRegressionEvaluationMeasure> createRegressionEvaluationMeasure() const {
                return this->createDecomposableRegressionLoss();
            }
    };

    DecomposableSquaredErrorLossConfig::DecomposableSquaredErrorLossConfig(
      ReadableProperty<IHeadConfig> headConfigGetter)
        : headConfig_(headConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      DecomposableSquaredErrorLossConfig::createStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                          const IRowWiseLabelMatrix& labelMatrix,
                                                                          const Blas& blas, const Lapack& lapack,
                                                                          bool preferSparseStatistics) const {
        return headConfig_.get().createStatisticsProviderFactory(featureMatrix, labelMatrix, *this);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      DecomposableSquaredErrorLossConfig::createStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
        const Lapack& lapack, bool preferSparseStatistics) const {
        return headConfigGetter_().createStatisticsProviderFactory(featureMatrix, regressionMatrix, *this);
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

    std::unique_ptr<IDecomposableClassificationLossFactory>
      DecomposableSquaredErrorLossConfig::createDecomposableClassificationLossFactory() const {
        return std::make_unique<DecomposableSquaredErrorLossFactory>();
    }

    std::unique_ptr<IDecomposableRegressionLossFactory>
      DecomposableSquaredErrorLossConfig::createDecomposableRegressionLossFactory() const {
        return std::make_unique<DecomposableSquaredErrorLossFactory>();
    }

}
