#include "mlrl/boosting/losses/loss_non_decomposable_squared_error.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse.hpp"
#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/util/iterators.hpp"
#include "mlrl/common/util/math.hpp"

#include <functional>

namespace boosting {

    template<typename GroundTruthIterator>
    using GroundTruthConversionFunction = std::function<float32(typename util::iterator_value<GroundTruthIterator>)>;

    template<typename ScoreIterator, typename GroundTruthIterator, typename StatisticIterator>
    static inline void updateDecomposableStatisticsInternally(
      ScoreIterator scoreIterator, GroundTruthIterator groundTruthIterator, StatisticIterator statisticIterator,
      uint32 numOutputs, GroundTruthConversionFunction<GroundTruthIterator> groundTruthConversionFunction) {
        typedef typename util::iterator_value<ScoreIterator> statistic_type;
        typedef typename util::iterator_value<GroundTruthIterator> ground_truth_type;
        GroundTruthIterator groundTruthIterator2 = groundTruthIterator;

        // For each output `i`, calculate `x_i = predictedScore_i^2 + (-2 * expectedScore_i * predictedScore_i) + 1` and
        // sum up those values. The sum is used as a denominator when calculating the gradients and Hessians
        // afterwards...
        statistic_type denominator = 0;

        for (uint32 i = 0; i < numOutputs; i++) {
            statistic_type predictedScore = scoreIterator[i];
            ground_truth_type groundTruth = *groundTruthIterator;
            statistic_type expectedScore = groundTruthConversionFunction(groundTruth);
            statistic_type x = (predictedScore * predictedScore) + (-2 * expectedScore * predictedScore) + 1;
            statisticIterator[i].gradient = x;  // Temporarily store `x` in the array of gradients
            denominator += x;
            groundTruthIterator++;
        }

        // The denominator that is used for the calculation of gradients is `sqrt(x_1 + x_2 + ...)`...
        statistic_type denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        statistic_type denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numOutputs; i++) {
            statistic_type predictedScore = scoreIterator[i];
            ground_truth_type groundTruth = *groundTruthIterator2;
            statistic_type expectedScore = groundTruthConversionFunction(groundTruth);
            Statistic<statistic_type>& statistic = statisticIterator[i];
            statistic_type x = statistic.gradient;

            // Calculate the gradient as `(predictedScore_i - expectedScore_i) / sqrt(x_1 + x_2 + ...)`...
            statistic.gradient = util::divideOrZero(predictedScore - expectedScore, denominatorGradient);

            // Calculate the Hessian on the diagonal of the Hessian matrix as
            // `(x_1 + ... + x_i-1 + x_i+1 + ...) / (x_1 + x_2 + ...)^1.5`...
            statistic.hessian = util::divideOrZero(denominator - x, denominatorHessian);
            groundTruthIterator2++;
        }
    }

    template<typename ScoreIterator, typename GroundTruthIterator, typename GradientIterator, typename HessianIterator>
    static inline void updateNonDecomposableStatisticsInternally(
      ScoreIterator scoreIterator, GroundTruthIterator groundTruthIterator, GradientIterator gradientIterator,
      HessianIterator hessianIterator, uint32 numOutputs,
      GroundTruthConversionFunction<GroundTruthIterator> groundTruthConversionFunction) {
        typedef typename util::iterator_value<ScoreIterator> statistic_type;
        typedef typename util::iterator_value<GroundTruthIterator> ground_truth_type;
        GroundTruthIterator groundTruthIterator2 = groundTruthIterator;
        GroundTruthIterator groundTruthIterator3 = groundTruthIterator;

        // For each output `i`, calculate `x_i = predictedScore_i^2 + (-2 * expectedScore_i * predictedScore_i) + 1` and
        // sum up those values. The sum is used as a denominator when calculating the gradients and Hessians
        // afterwards...
        statistic_type denominator = 0;

        for (uint32 i = 0; i < numOutputs; i++) {
            statistic_type predictedScore = scoreIterator[i];
            ground_truth_type groundTruth = *groundTruthIterator;
            statistic_type expectedScore = groundTruthConversionFunction(groundTruth);
            statistic_type x = (predictedScore * predictedScore) + (-2 * expectedScore * predictedScore) + 1;
            gradientIterator[i] = x;  // Temporarily store `x` in the array of gradients
            denominator += x;
            groundTruthIterator++;
        }

        // The denominator that is used for the calculation of gradients is `sqrt(x_1 + x_2 + ...)`...
        statistic_type denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        statistic_type denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numOutputs; i++) {
            statistic_type predictedScore = scoreIterator[i];
            ground_truth_type groundTruth = *groundTruthIterator2;
            statistic_type expectedScore = groundTruthConversionFunction(groundTruth);
            statistic_type x = gradientIterator[i];

            // Calculate the Hessians that belong to the part of the Hessian matrix' upper triangle that corresponds to
            // the current output. Such a hessian calculates as
            // `-(predictedScore_i - expectedScore_i) * (predictedScore_j - expectedScore_j) / (x_1 + x_2 + ...)^1.5`
            GroundTruthIterator groundTruthIterator4 = groundTruthIterator3;

            for (uint32 j = 0; j < i; j++) {
                statistic_type predictedScore2 = scoreIterator[j];
                ground_truth_type groundTruth2 = *groundTruthIterator4;
                statistic_type expectedScore2 = groundTruthConversionFunction(groundTruth2);
                *hessianIterator = util::divideOrZero(
                  -(predictedScore - expectedScore) * (predictedScore2 - expectedScore2), denominatorHessian);
                hessianIterator++;
                groundTruthIterator4++;
            }

            // Calculate the gradient as `(predictedScore_i - expectedScore_i) / sqrt(x_1 + x_2 + ...)`...
            gradientIterator[i] = util::divideOrZero(predictedScore - expectedScore, denominatorGradient);

            // Calculate the Hessian on the diagonal of the Hessian matrix as
            // `(x_1 + ... + x_i-1 + x_i+1 + ...) / (x_1 + x_2 + ...)^1.5`...
            *hessianIterator = util::divideOrZero(denominator - x, denominatorHessian);
            hessianIterator++;
            groundTruthIterator2++;
        }
    }

    template<typename ScoreIterator, typename GroundTruthIterator>
    static inline typename util::iterator_value<ScoreIterator> evaluateInternally(
      ScoreIterator scoreIterator, GroundTruthIterator groundTruthIterator, uint32 numOutputs,
      GroundTruthConversionFunction<GroundTruthIterator> groundTruthConversionFunction) {
        typedef typename util::iterator_value<ScoreIterator> score_type;

        // The example-wise squared error loss calculates as `sqrt((expectedScore_1 - predictedScore_1)^2 + ...)`.
        score_type sumOfSquares = 0;

        for (uint32 i = 0; i < numOutputs; i++) {
            score_type predictedScore = scoreIterator[i];
            bool trueLabel = *groundTruthIterator;
            score_type expectedScore = trueLabel ? 1 : -1;
            score_type difference = (expectedScore - predictedScore);
            sumOfSquares += (difference * difference);
            groundTruthIterator++;
        }

        return std::sqrt(sumOfSquares);
    }

    static inline constexpr float32 binaryConversionFunction(bool groundTruth) {
        return groundTruth ? 1.0f : -1.0f;
    }

    static inline constexpr float32 scoreConversionFunction(float32 groundTruth) {
        return groundTruth;
    }

    /**
     * An implementation of the type `INonDecomposableClassificationLoss` that implements a multivariate variant of the
     * squared error loss that is non-decomposable.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class NonDecomposableSquaredErrorLoss final : public INonDecomposableClassificationLoss<StatisticType>,
                                                  public INonDecomposableRegressionLoss<StatisticType> {
        public:

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                updateDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.values_begin(exampleIndex), labelMatrix.numCols, &binaryConversionFunction);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                updateDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.values_begin(exampleIndex), labelMatrix.numCols, &binaryConversionFunction);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                auto groundTruthIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                             labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols,
                                                       &binaryConversionFunction);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              PartialIndexVector::const_iterator indicesBegin, PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                auto groundTruthIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                             labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols,
                                                       &binaryConversionFunction);
            }

            void updateDecomposableStatistics(uint32 exampleIndex,
                                              const CContiguousView<const float32>& regressionMatrix,
                                              const CContiguousView<StatisticType>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                updateDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cbegin(exampleIndex),
                  statisticView.values_begin(exampleIndex), regressionMatrix.numCols, &scoreConversionFunction);
            }

            void updateDecomposableStatistics(uint32 exampleIndex,
                                              const CContiguousView<const float32>& regressionMatrix,
                                              const CContiguousView<StatisticType>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                updateDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cbegin(exampleIndex),
                  statisticView.values_begin(exampleIndex), regressionMatrix.numCols, &binaryConversionFunction);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                                              const CContiguousView<StatisticType>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                auto groundTruthIterator = createSparseForwardIterator(
                  regressionMatrix.indices_cbegin(exampleIndex), regressionMatrix.indices_cend(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                       statisticView.values_begin(exampleIndex),
                                                       regressionMatrix.numCols, &scoreConversionFunction);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                                              const CContiguousView<StatisticType>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                auto groundTruthIterator = createSparseForwardIterator(
                  regressionMatrix.indices_cbegin(exampleIndex), regressionMatrix.indices_cend(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                       statisticView.values_begin(exampleIndex),
                                                       regressionMatrix.numCols, &scoreConversionFunction);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const override {
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.gradients_begin(exampleIndex), statisticView.hessians_begin(exampleIndex),
                  labelMatrix.numCols, &binaryConversionFunction);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const override {
                auto groundTruthIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                             labelMatrix.indices_cend(exampleIndex));
                updateNonDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                          statisticView.gradients_begin(exampleIndex),
                                                          statisticView.hessians_begin(exampleIndex),
                                                          labelMatrix.numCols, &binaryConversionFunction);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const override {
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cbegin(exampleIndex),
                  statisticView.gradients_begin(exampleIndex), statisticView.hessians_begin(exampleIndex),
                  regressionMatrix.numCols, &scoreConversionFunction);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const override {
                auto groundTruthIterator = createSparseForwardIterator(
                  regressionMatrix.indices_cbegin(exampleIndex), regressionMatrix.indices_cend(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cend(exampleIndex));
                updateNonDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                          statisticView.gradients_begin(exampleIndex),
                                                          statisticView.hessians_begin(exampleIndex),
                                                          regressionMatrix.numCols, &scoreConversionFunction);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          labelMatrix.values_cbegin(exampleIndex), labelMatrix.numCols,
                                          &binaryConversionFunction);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override {
                auto groundTruthIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                             labelMatrix.indices_cend(exampleIndex));
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                          labelMatrix.numCols, &binaryConversionFunction);
            }

            /**
             * @see `IRegressionEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.numCols,
                                          &scoreConversionFunction);
            }

            /**
             * @see `IRegressionEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override {
                auto groundTruthIterator = createSparseForwardIterator(
                  regressionMatrix.indices_cbegin(exampleIndex), regressionMatrix.indices_cend(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cend(exampleIndex));
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                          regressionMatrix.numCols, &scoreConversionFunction);
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            StatisticType measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                          typename View<StatisticType>::const_iterator scoresBegin,
                                          typename View<StatisticType>::const_iterator scoresEnd) const override {
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = createBinarySparseForwardIterator(labelVector.cbegin(), labelVector.cend());
                return evaluateInternally(scoresBegin, labelIterator, numLabels, &binaryConversionFunction);
            }
    };

    /**
     * Allows to create instances of the type `INonDecomposableClassificationLoss` that implement a multivariate variant
     * of the squared error loss that is non-decomposable.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class NonDecomposableSquaredErrorLossFactory final
        : public INonDecomposableClassificationLossFactory<StatisticType>,
          public INonDecomposableRegressionLossFactory<StatisticType> {
        public:

            std::unique_ptr<INonDecomposableClassificationLoss<StatisticType>> createNonDecomposableClassificationLoss()
              const override {
                return std::make_unique<NonDecomposableSquaredErrorLoss<StatisticType>>();
            }

            std::unique_ptr<INonDecomposableRegressionLoss<StatisticType>> createNonDecomposableRegressionLoss()
              const override {
                return std::make_unique<NonDecomposableSquaredErrorLoss<StatisticType>>();
            }

            std::unique_ptr<IDistanceMeasure<StatisticType>> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const override {
                return this->createNonDecomposableClassificationLoss();
            }

            std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> createClassificationEvaluationMeasure()
              const override {
                return this->createNonDecomposableClassificationLoss();
            }

            std::unique_ptr<IRegressionEvaluationMeasure<StatisticType>> createRegressionEvaluationMeasure()
              const override {
                return this->createNonDecomposableRegressionLoss();
            }
    };

    NonDecomposableSquaredErrorLossConfig::NonDecomposableSquaredErrorLossConfig(
      ReadableProperty<IStatisticTypeConfig> statisticTypeConfig)
        : statisticTypeConfig_(statisticTypeConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      NonDecomposableSquaredErrorLossConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return statisticTypeConfig_.get().createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, *this, blasFactory, lapackFactory);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      NonDecomposableSquaredErrorLossConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return statisticTypeConfig_.get().createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix,
                                                                                    *this, blasFactory, lapackFactory);
    }

    std::unique_ptr<IMarginalProbabilityFunctionFactory>
      NonDecomposableSquaredErrorLossConfig::createMarginalProbabilityFunctionFactory() const {
        return nullptr;
    }

    std::unique_ptr<IJointProbabilityFunctionFactory>
      NonDecomposableSquaredErrorLossConfig::createJointProbabilityFunctionFactory() const {
        return nullptr;
    }

    float64 NonDecomposableSquaredErrorLossConfig::getDefaultPrediction() const {
        return 0.0;
    }

    std::unique_ptr<INonDecomposableClassificationLossFactory<float64>>
      NonDecomposableSquaredErrorLossConfig::createNonDecomposableClassificationLossFactory() const {
        return std::make_unique<NonDecomposableSquaredErrorLossFactory<float64>>();
    }

    std::unique_ptr<INonDecomposableRegressionLossFactory<float64>>
      NonDecomposableSquaredErrorLossConfig::createNonDecomposableRegressionLossFactory() const {
        return std::make_unique<NonDecomposableSquaredErrorLossFactory<float64>>();
    }

}
