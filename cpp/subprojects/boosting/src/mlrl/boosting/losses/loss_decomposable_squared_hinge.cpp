#include "mlrl/boosting/losses/loss_decomposable_squared_hinge.hpp"

#include "loss_decomposable_sparse_common.hpp"

namespace boosting {

    template<typename StatisticType>
    static inline void updateGradientAndHessian(bool trueLabel, StatisticType predictedScore, StatisticType& gradient,
                                                StatisticType& hessian) {
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

    template<typename ScoreType>
    static inline ScoreType evaluatePrediction(bool trueLabel, ScoreType predictedScore) {
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
     * Allows to create instances of the type `IDecomposableClassificationLoss` that implement a multivariate variant of
     * the squared hinge loss that is decomposable.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class DecomposableSquaredHingeLossFactory final
        : public ISparseDecomposableClassificationLossFactory<StatisticType> {
        public:

            std::unique_ptr<ISparseDecomposableClassificationLoss<StatisticType>>
              createSparseDecomposableClassificationLoss() const override {
                return std::make_unique<SparseDecomposableClassificationLoss<StatisticType>>(
                  &updateGradientAndHessian<StatisticType>, &evaluatePrediction<StatisticType>);
            }

            std::unique_ptr<IDistanceMeasure<StatisticType>> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const override {
                return this->createSparseDecomposableClassificationLoss();
            }
    };

    DecomposableSquaredHingeLossConfig::DecomposableSquaredHingeLossConfig(
      ReadableProperty<IStatisticTypeConfig> statisticTypeConfig)
        : statisticTypeConfig_(statisticTypeConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      DecomposableSquaredHingeLossConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        if (preferSparseStatistics) {
            return statisticTypeConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix,
                                                                                            *this);
        } else {
            return statisticTypeConfig_.get().createClassificationStatisticsProviderFactory(
              featureMatrix, labelMatrix, static_cast<const IDecomposableClassificationLossConfig&>(*this));
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

    std::unique_ptr<ISparseDecomposableClassificationLossFactory<float64>>
      DecomposableSquaredHingeLossConfig::createSparseDecomposableClassificationLossFactory() const {
        return std::make_unique<DecomposableSquaredHingeLossFactory<float64>>();
    }

}
