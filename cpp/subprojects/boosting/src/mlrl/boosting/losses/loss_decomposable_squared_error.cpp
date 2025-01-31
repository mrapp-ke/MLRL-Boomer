#include "mlrl/boosting/losses/loss_decomposable_squared_error.hpp"

#include "loss_decomposable_common.hpp"

namespace boosting {

    template<typename StatisticType>
    static inline void updateGradientAndHessianRegression(float32 expectedScore, StatisticType predictedScore,
                                                          StatisticType& gradient, StatisticType& hessian) {
        gradient = (predictedScore - (StatisticType) expectedScore);
        hessian = 1;
    }

    template<typename StatisticType>
    static inline void updateGradientAndHessianClassification(bool trueLabel, StatisticType predictedScore,
                                                              StatisticType& gradient, StatisticType& hessian) {
        updateGradientAndHessianRegression(trueLabel ? 1.0f : -1.0f, predictedScore, gradient, hessian);
    }

    template<typename ScoreType>
    static inline ScoreType evaluatePredictionRegression(float32 expectedScore, ScoreType predictedScore) {
        ScoreType difference = ((ScoreType) expectedScore - predictedScore);
        return difference * difference;
    }

    template<typename ScoreType>
    static inline ScoreType evaluatePredictionClassification(bool trueLabel, ScoreType predictedScore) {
        return evaluatePredictionRegression(trueLabel ? 1.0 : -1.0, predictedScore);
    }

    /**
     * Allows to create instances of the type `IDecomposableClassificationLoss` that implement a multivariate variant of
     * the squared error loss that is decomposable.
     */
    class DecomposableSquaredErrorLossFactory final : public IDecomposableClassificationLossFactory<float64>,
                                                      public IDecomposableRegressionLossFactory<float64> {
        public:

            std::unique_ptr<IDecomposableClassificationLoss<float64>> createDecomposableClassificationLoss()
              const override {
                return std::make_unique<DecomposableClassificationLoss<float64>>(
                  &updateGradientAndHessianClassification<float64>, &evaluatePredictionClassification<float64>);
            }

            std::unique_ptr<IDecomposableRegressionLoss<float64>> createDecomposableRegressionLoss() const override {
                return std::make_unique<DecomposableRegressionLoss<float64>>(
                  &updateGradientAndHessianRegression<float64>, &evaluatePredictionRegression<float64>);
            }

            std::unique_ptr<IDistanceMeasure<float64>> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const override {
                return this->createDecomposableClassificationLoss();
            }

            std::unique_ptr<IClassificationEvaluationMeasure<float64>> createClassificationEvaluationMeasure()
              const override {
                return this->createDecomposableClassificationLoss();
            }

            std::unique_ptr<IRegressionEvaluationMeasure<float64>> createRegressionEvaluationMeasure() const override {
                return this->createDecomposableRegressionLoss();
            }
    };

    DecomposableSquaredErrorLossConfig::DecomposableSquaredErrorLossConfig(ReadableProperty<IHeadConfig> headConfig)
        : headConfig_(headConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      DecomposableSquaredErrorLossConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return headConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, *this);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      DecomposableSquaredErrorLossConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return headConfig_.get().createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix, *this);
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

    std::unique_ptr<IDecomposableClassificationLossFactory<float64>>
      DecomposableSquaredErrorLossConfig::createDecomposableClassificationLossFactory() const {
        return std::make_unique<DecomposableSquaredErrorLossFactory>();
    }

    std::unique_ptr<IDecomposableRegressionLossFactory<float64>>
      DecomposableSquaredErrorLossConfig::createDecomposableRegressionLossFactory() const {
        return std::make_unique<DecomposableSquaredErrorLossFactory>();
    }

}
