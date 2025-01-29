#include "mlrl/boosting/losses/loss_non_decomposable_squared_error.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse.hpp"
#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/util/iterators.hpp"
#include "mlrl/common/util/math.hpp"

#include <functional>

namespace boosting {

    template<typename GroundTruthIterator>
    using GroundTruthConversionFunction = std::function<float32(typename util::iterator_value<GroundTruthIterator>)>;

    template<typename GroundTruthIterator>
    static inline void updateDecomposableStatisticsInternally(
      View<float64>::const_iterator scoreIterator, GroundTruthIterator groundTruthIterator,
      View<Statistic<float64>>::iterator statisticIterator, uint32 numOutputs,
      GroundTruthConversionFunction<GroundTruthIterator> groundTruthConversionFunction) {
        typedef typename util::iterator_value<GroundTruthIterator> ground_truth_type;
        GroundTruthIterator groundTruthIterator2 = groundTruthIterator;

        // For each output `i`, calculate `x_i = predictedScore_i^2 + (-2 * expectedScore_i * predictedScore_i) + 1` and
        // sum up those values. The sum is used as a denominator when calculating the gradients and Hessians
        // afterwards...
        float64 denominator = 0;

        for (uint32 i = 0; i < numOutputs; i++) {
            float64 predictedScore = scoreIterator[i];
            ground_truth_type groundTruth = *groundTruthIterator;
            float64 expectedScore = groundTruthConversionFunction(groundTruth);
            float64 x = (predictedScore * predictedScore) + (-2 * expectedScore * predictedScore) + 1;
            statisticIterator[i].gradient = x;  // Temporarily store `x` in the array of gradients
            denominator += x;
            groundTruthIterator++;
        }

        // The denominator that is used for the calculation of gradients is `sqrt(x_1 + x_2 + ...)`...
        float64 denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        float64 denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numOutputs; i++) {
            float64 predictedScore = scoreIterator[i];
            ground_truth_type groundTruth = *groundTruthIterator2;
            float64 expectedScore = groundTruthConversionFunction(groundTruth);
            Statistic<float64>& statistic = statisticIterator[i];
            float64 x = statistic.gradient;

            // Calculate the gradient as `(predictedScore_i - expectedScore_i) / sqrt(x_1 + x_2 + ...)`...
            statistic.gradient = util::divideOrZero<float64>(predictedScore - expectedScore, denominatorGradient);

            // Calculate the Hessian on the diagonal of the Hessian matrix as
            // `(x_1 + ... + x_i-1 + x_i+1 + ...) / (x_1 + x_2 + ...)^1.5`...
            statistic.hessian = util::divideOrZero<float64>(denominator - x, denominatorHessian);
            groundTruthIterator2++;
        }
    }

    template<typename GroundTruthIterator>
    static inline void updateNonDecomposableStatisticsInternally(
      View<float64>::const_iterator scoreIterator, GroundTruthIterator groundTruthIterator,
      DenseNonDecomposableStatisticView<float64>::gradient_iterator gradientIterator,
      DenseNonDecomposableStatisticView<float64>::hessian_iterator hessianIterator, uint32 numOutputs,
      GroundTruthConversionFunction<GroundTruthIterator> groundTruthConversionFunction) {
        typedef typename util::iterator_value<GroundTruthIterator> ground_truth_type;
        GroundTruthIterator groundTruthIterator2 = groundTruthIterator;
        GroundTruthIterator groundTruthIterator3 = groundTruthIterator;

        // For each output `i`, calculate `x_i = predictedScore_i^2 + (-2 * expectedScore_i * predictedScore_i) + 1` and
        // sum up those values. The sum is used as a denominator when calculating the gradients and Hessians
        // afterwards...
        float64 denominator = 0;

        for (uint32 i = 0; i < numOutputs; i++) {
            float64 predictedScore = scoreIterator[i];
            ground_truth_type groundTruth = *groundTruthIterator;
            float64 expectedScore = groundTruthConversionFunction(groundTruth);
            float64 x = (predictedScore * predictedScore) + (-2 * expectedScore * predictedScore) + 1;
            gradientIterator[i] = x;  // Temporarily store `x` in the array of gradients
            denominator += x;
            groundTruthIterator++;
        }

        // The denominator that is used for the calculation of gradients is `sqrt(x_1 + x_2 + ...)`...
        float64 denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        float64 denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numOutputs; i++) {
            float64 predictedScore = scoreIterator[i];
            ground_truth_type groundTruth = *groundTruthIterator2;
            float64 expectedScore = groundTruthConversionFunction(groundTruth);
            float64 x = gradientIterator[i];

            // Calculate the Hessians that belong to the part of the Hessian matrix' upper triangle that corresponds to
            // the current output. Such a hessian calculates as
            // `-(predictedScore_i - expectedScore_i) * (predictedScore_j - expectedScore_j) / (x_1 + x_2 + ...)^1.5`
            GroundTruthIterator groundTruthIterator4 = groundTruthIterator3;

            for (uint32 j = 0; j < i; j++) {
                float64 predictedScore2 = scoreIterator[j];
                ground_truth_type groundTruth2 = *groundTruthIterator4;
                float64 expectedScore2 = groundTruthConversionFunction(groundTruth2);
                *hessianIterator = util::divideOrZero<float64>(
                  -(predictedScore - expectedScore) * (predictedScore2 - expectedScore2), denominatorHessian);
                hessianIterator++;
                groundTruthIterator4++;
            }

            // Calculate the gradient as `(predictedScore_i - expectedScore_i) / sqrt(x_1 + x_2 + ...)`...
            gradientIterator[i] = util::divideOrZero<float64>(predictedScore - expectedScore, denominatorGradient);

            // Calculate the Hessian on the diagonal of the Hessian matrix as
            // `(x_1 + ... + x_i-1 + x_i+1 + ...) / (x_1 + x_2 + ...)^1.5`...
            *hessianIterator = util::divideOrZero<float64>(denominator - x, denominatorHessian);
            hessianIterator++;
            groundTruthIterator2++;
        }
    }

    template<typename GroundTruthIterator>
    static inline float64 evaluateInternally(
      View<float64>::const_iterator scoreIterator, GroundTruthIterator groundTruthIterator, uint32 numOutputs,
      GroundTruthConversionFunction<GroundTruthIterator> groundTruthConversionFunction) {
        // The example-wise squared error loss calculates as `sqrt((expectedScore_1 - predictedScore_1)^2 + ...)`.
        float64 sumOfSquares = 0;

        for (uint32 i = 0; i < numOutputs; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *groundTruthIterator;
            float64 expectedScore = trueLabel ? 1 : -1;
            float64 difference = (expectedScore - predictedScore);
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
     */
    class NonDecomposableSquaredErrorLoss final : public INonDecomposableClassificationLoss<float64>,
                                                  public INonDecomposableRegressionLoss<float64> {
        public:

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<float64>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<float64>>& statisticView) const override {
                updateDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.values_begin(exampleIndex), labelMatrix.numCols, &binaryConversionFunction);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<float64>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<float64>>& statisticView) const override {
                updateDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.values_begin(exampleIndex), labelMatrix.numCols, &binaryConversionFunction);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<float64>& scoreMatrix,
              CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<float64>>& statisticView) const override {
                auto groundTruthIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                             labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols,
                                                       &binaryConversionFunction);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<float64>& scoreMatrix,
              PartialIndexVector::const_iterator indicesBegin, PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<float64>>& statisticView) const override {
                auto groundTruthIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                             labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols,
                                                       &binaryConversionFunction);
            }

            void updateDecomposableStatistics(uint32 exampleIndex,
                                              const CContiguousView<const float32>& regressionMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override {
                updateDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cbegin(exampleIndex),
                  statisticView.values_begin(exampleIndex), regressionMatrix.numCols, &scoreConversionFunction);
            }

            void updateDecomposableStatistics(uint32 exampleIndex,
                                              const CContiguousView<const float32>& regressionMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override {
                updateDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cbegin(exampleIndex),
                  statisticView.values_begin(exampleIndex), regressionMatrix.numCols, &binaryConversionFunction);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override {
                auto groundTruthIterator = createSparseForwardIterator(
                  regressionMatrix.indices_cbegin(exampleIndex), regressionMatrix.indices_cend(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                       statisticView.values_begin(exampleIndex),
                                                       regressionMatrix.numCols, &scoreConversionFunction);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override {
                auto groundTruthIterator = createSparseForwardIterator(
                  regressionMatrix.indices_cbegin(exampleIndex), regressionMatrix.indices_cend(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                       statisticView.values_begin(exampleIndex),
                                                       regressionMatrix.numCols, &scoreConversionFunction);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<float64>& scoreMatrix,
              DenseNonDecomposableStatisticView<float64>& statisticView) const override {
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.gradients_begin(exampleIndex), statisticView.hessians_begin(exampleIndex),
                  labelMatrix.numCols, &binaryConversionFunction);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<float64>& scoreMatrix,
              DenseNonDecomposableStatisticView<float64>& statisticView) const override {
                auto groundTruthIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                             labelMatrix.indices_cend(exampleIndex));
                updateNonDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                                          statisticView.gradients_begin(exampleIndex),
                                                          statisticView.hessians_begin(exampleIndex),
                                                          labelMatrix.numCols, &binaryConversionFunction);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
              const CContiguousView<float64>& scoreMatrix,
              DenseNonDecomposableStatisticView<float64>& statisticView) const override {
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cbegin(exampleIndex),
                  statisticView.gradients_begin(exampleIndex), statisticView.hessians_begin(exampleIndex),
                  regressionMatrix.numCols, &scoreConversionFunction);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
              const CContiguousView<float64>& scoreMatrix,
              DenseNonDecomposableStatisticView<float64>& statisticView) const override {
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
            float64 evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          labelMatrix.values_cbegin(exampleIndex), labelMatrix.numCols,
                                          &binaryConversionFunction);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override {
                auto groundTruthIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                             labelMatrix.indices_cend(exampleIndex));
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                          labelMatrix.numCols, &binaryConversionFunction);
            }

            /**
             * @see `IRegressionEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.numCols,
                                          &scoreConversionFunction);
            }

            /**
             * @see `IRegressionEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override {
                auto groundTruthIterator = createSparseForwardIterator(
                  regressionMatrix.indices_cbegin(exampleIndex), regressionMatrix.indices_cend(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cend(exampleIndex));
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex), groundTruthIterator,
                                          regressionMatrix.numCols, &scoreConversionFunction);
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            float64 measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                    View<float64>::const_iterator scoresBegin,
                                    View<float64>::const_iterator scoresEnd) const override {
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = createBinarySparseForwardIterator(labelVector.cbegin(), labelVector.cend());
                return evaluateInternally(scoresBegin, labelIterator, numLabels, &binaryConversionFunction);
            }
    };

    /**
     * Allows to create instances of the type `INonDecomposableClassificationLoss` that implement a multivariate variant
     * of the squared error loss that is non-decomposable.
     */
    class NonDecomposableSquaredErrorLossFactory final : public INonDecomposableClassificationLossFactory,
                                                         public INonDecomposableRegressionLossFactory {
        public:

            std::unique_ptr<INonDecomposableClassificationLoss<float64>> createNonDecomposableClassificationLoss()
              const override {
                return std::make_unique<NonDecomposableSquaredErrorLoss>();
            }

            std::unique_ptr<INonDecomposableRegressionLoss<float64>> createNonDecomposableRegressionLoss()
              const override {
                return std::make_unique<NonDecomposableSquaredErrorLoss>();
            }

            std::unique_ptr<IDistanceMeasure<float64>> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const override {
                return this->createNonDecomposableClassificationLoss();
            }

            std::unique_ptr<IClassificationEvaluationMeasure> createClassificationEvaluationMeasure() const override {
                return this->createNonDecomposableClassificationLoss();
            }

            std::unique_ptr<IRegressionEvaluationMeasure> createRegressionEvaluationMeasure() const override {
                return this->createNonDecomposableRegressionLoss();
            }
    };

    NonDecomposableSquaredErrorLossConfig::NonDecomposableSquaredErrorLossConfig(
      ReadableProperty<IHeadConfig> headConfig)
        : headConfig_(headConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      NonDecomposableSquaredErrorLossConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return headConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, *this,
                                                                               blasFactory, lapackFactory);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      NonDecomposableSquaredErrorLossConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return headConfig_.get().createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix, *this,
                                                                           blasFactory, lapackFactory);
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

    std::unique_ptr<INonDecomposableClassificationLossFactory>
      NonDecomposableSquaredErrorLossConfig::createNonDecomposableClassificationLossFactory() const {
        return std::make_unique<NonDecomposableSquaredErrorLossFactory>();
    }

    std::unique_ptr<INonDecomposableRegressionLossFactory>
      NonDecomposableSquaredErrorLossConfig::createNonDecomposableRegressionLossFactory() const {
        return std::make_unique<NonDecomposableSquaredErrorLossFactory>();
    }

}
