#include "mlrl/boosting/losses/loss_non_decomposable_squared_error.hpp"

#include "mlrl/common/iterator/binary_forward_iterator.hpp"
#include "mlrl/common/util/math.hpp"

namespace boosting {

    template<typename LabelIterator>
    static inline void updateDecomposableStatisticsInternally(View<float64>::const_iterator scoreIterator,
                                                              LabelIterator labelIterator,
                                                              View<Tuple<float64>>::iterator statisticIterator,
                                                              uint32 numLabels) {
        LabelIterator labelIterator2 = labelIterator;

        // For each label `i`, calculate `x_i = predictedScore_i^2 + (-2 * expectedScore_i * predictedScore_i) + 1` and
        // sum up those values. The sum is used as a denominator when calculating the gradients and Hessians
        // afterwards...
        float64 denominator = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;
            float64 expectedScore = trueLabel ? 1 : -1;
            float64 x = (predictedScore * predictedScore) + (-2 * expectedScore * predictedScore) + 1;
            statisticIterator[i].first = x;  // Temporarily store `x` in the array of gradients
            denominator += x;
            labelIterator++;
        }

        // The denominator that is used for the calculation of gradients is `sqrt(x_1 + x_2 + ...)`...
        float64 denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        float64 denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator2;
            float64 expectedScore = trueLabel ? 1 : -1;
            Tuple<float64>& tuple = statisticIterator[i];
            float64 x = tuple.first;

            // Calculate the gradient as `(predictedScore_i - expectedScore_i) / sqrt(x_1 + x_2 + ...)`...
            tuple.first = divideOrZero<float64>(predictedScore - expectedScore, denominatorGradient);

            // Calculate the Hessian on the diagonal of the Hessian matrix as
            // `(x_1 + ... + x_i-1 + x_i+1 + ...) / (x_1 + x_2 + ...)^1.5`...
            tuple.second = divideOrZero<float64>(denominator - x, denominatorHessian);
            labelIterator2++;
        }
    }

    template<typename LabelIterator>
    static inline void updateNonDecomposableStatisticsInternally(
      View<float64>::const_iterator scoreIterator, LabelIterator labelIterator,
      DenseNonDecomposableStatisticView::gradient_iterator gradientIterator,
      DenseNonDecomposableStatisticView::hessian_iterator hessianIterator, uint32 numLabels) {
        LabelIterator labelIterator2 = labelIterator;
        LabelIterator labelIterator3 = labelIterator;

        // For each label `i`, calculate `x_i = predictedScore_i^2 + (-2 * expectedScore_i * predictedScore_i) + 1` and
        // sum up those values. The sum is used as a denominator when calculating the gradients and Hessians
        // afterwards...
        float64 denominator = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;
            float64 expectedScore = trueLabel ? 1 : -1;
            float64 x = (predictedScore * predictedScore) + (-2 * expectedScore * predictedScore) + 1;
            gradientIterator[i] = x;  // Temporarily store `x` in the array of gradients
            denominator += x;
            labelIterator++;
        }

        // The denominator that is used for the calculation of gradients is `sqrt(x_1 + x_2 + ...)`...
        float64 denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        float64 denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator2;
            float64 expectedScore = trueLabel ? 1 : -1;
            float64 x = gradientIterator[i];

            // Calculate the Hessians that belong to the part of the Hessian matrix' upper triangle that corresponds to
            // the current label. Such a hessian calculates as
            // `-(predictedScore_i - expectedScore_i) * (predictedScore_j - expectedScore_j) / (x_1 + x_2 + ...)^1.5`
            LabelIterator labelIterator4 = labelIterator3;

            for (uint32 j = 0; j < i; j++) {
                float64 predictedScore2 = scoreIterator[j];
                bool trueLabel2 = *labelIterator4;
                float64 expectedScore2 = trueLabel2 ? 1 : -1;
                *hessianIterator = divideOrZero<float64>(
                  -(predictedScore - expectedScore) * (predictedScore2 - expectedScore2), denominatorHessian);
                hessianIterator++;
                labelIterator4++;
            }

            // Calculate the gradient as `(predictedScore_i - expectedScore_i) / sqrt(x_1 + x_2 + ...)`...
            gradientIterator[i] = divideOrZero<float64>(predictedScore - expectedScore, denominatorGradient);

            // Calculate the Hessian on the diagonal of the Hessian matrix as
            // `(x_1 + ... + x_i-1 + x_i+1 + ...) / (x_1 + x_2 + ...)^1.5`...
            *hessianIterator = divideOrZero<float64>(denominator - x, denominatorHessian);
            hessianIterator++;
            labelIterator2++;
        }
    }

    template<typename LabelIterator>
    static inline float64 evaluateInternally(View<float64>::const_iterator scoreIterator, LabelIterator labelIterator,
                                             uint32 numLabels) {
        // The example-wise squared error loss calculates as `sqrt((expectedScore_1 - predictedScore_1)^2 + ...)`.
        float64 sumOfSquares = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;
            float64 expectedScore = trueLabel ? 1 : -1;
            float64 difference = (expectedScore - predictedScore);
            sumOfSquares += (difference * difference);
            labelIterator++;
        }

        return std::sqrt(sumOfSquares);
    }

    /**
     * An implementation of the type `INonDecomposableClassificationLoss` that implements a multivariate variant of the
     * squared error loss that is non-decomposable.
     */
    class NonDecomposableSquaredErrorLoss final : public INonDecomposableClassificationLoss {
        public:

            virtual void updateDecomposableStatistics(uint32 exampleIndex,
                                                      const CContiguousView<const uint8>& labelMatrix,
                                                      const CContiguousView<float64>& scoreMatrix,
                                                      CompleteIndexVector::const_iterator indicesBegin,
                                                      CompleteIndexVector::const_iterator indicesEnd,
                                                      CContiguousView<Tuple<float64>>& statisticView) const override {
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex),
                                                       labelMatrix.values_cbegin(exampleIndex),
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            virtual void updateDecomposableStatistics(uint32 exampleIndex,
                                                      const CContiguousView<const uint8>& labelMatrix,
                                                      const CContiguousView<float64>& scoreMatrix,
                                                      PartialIndexVector::const_iterator indicesBegin,
                                                      PartialIndexVector::const_iterator indicesEnd,
                                                      CContiguousView<Tuple<float64>>& statisticView) const override {
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex),
                                                       labelMatrix.values_cbegin(exampleIndex),
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            virtual void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                      const CContiguousView<float64>& scoreMatrix,
                                                      CompleteIndexVector::const_iterator indicesBegin,
                                                      CompleteIndexVector::const_iterator indicesEnd,
                                                      CContiguousView<Tuple<float64>>& statisticView) const override {
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            virtual void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                      const CContiguousView<float64>& scoreMatrix,
                                                      PartialIndexVector::const_iterator indicesBegin,
                                                      PartialIndexVector::const_iterator indicesEnd,
                                                      CContiguousView<Tuple<float64>>& statisticView) const override {
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            void updateNonDecomposableStatistics(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                                 const CContiguousView<float64>& scoreMatrix,
                                                 DenseNonDecomposableStatisticView& statisticView) const override {
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.gradients_begin(exampleIndex), statisticView.hessians_begin(exampleIndex),
                  labelMatrix.numCols);
            }

            void updateNonDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                 const CContiguousView<float64>& scoreMatrix,
                                                 DenseNonDecomposableStatisticView& statisticView) const override {
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelIterator, statisticView.gradients_begin(exampleIndex),
                  statisticView.hessians_begin(exampleIndex), labelMatrix.numCols);
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          labelMatrix.values_cbegin(exampleIndex), labelMatrix.numCols);
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override {
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator, labelMatrix.numCols);
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            float64 measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                    View<float64>::const_iterator scoresBegin,
                                    View<float64>::const_iterator scoresEnd) const override {
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());
                return evaluateInternally(scoresBegin, labelIterator, numLabels);
            }
    };

    /**
     * Allows to create instances of the type `INonDecomposableClassificationLoss` that implement a multivariate variant
     * of the squared error loss that is non-decomposable.
     */
    class NonDecomposableSquaredErrorLossFactory final : public INonDecomposableClassificationLossFactory {
        public:

            std::unique_ptr<INonDecomposableClassificationLoss> createNonDecomposableClassificationLoss()
              const override {
                return std::make_unique<NonDecomposableSquaredErrorLoss>();
            }

            std::unique_ptr<IDistanceMeasure> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const {
                return this->createNonDecomposableClassificationLoss();
            }

            std::unique_ptr<IEvaluationMeasure> createEvaluationMeasure() const {
                return this->createNonDecomposableClassificationLoss();
            }
    };

    NonDecomposableSquaredErrorLossConfig::NonDecomposableSquaredErrorLossConfig(
      ReadableProperty<IHeadConfig> headConfigGetter)
        : headConfig_(headConfigGetter) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      NonDecomposableSquaredErrorLossConfig::createStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                                             const IRowWiseLabelMatrix& labelMatrix,
                                                                             const Blas& blas, const Lapack& lapack,
                                                                             bool preferSparseStatistics) const {
        return headConfig_.get().createStatisticsProviderFactory(featureMatrix, labelMatrix, *this, blas, lapack);
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

}
