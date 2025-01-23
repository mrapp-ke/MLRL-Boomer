#include "mlrl/boosting/losses/loss_non_decomposable_squared_hinge.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/util/math.hpp"

namespace boosting {

    template<typename LabelIterator>
    static inline void updateDecomposableStatisticsInternally(View<float64>::const_iterator scoreIterator,
                                                              LabelIterator labelIterator,
                                                              View<Statistic<float64>>::iterator statisticIterator,
                                                              uint32 numLabels) {
        LabelIterator labelIterator2 = labelIterator;

        // For each label `i`, calculate `x_i = predictedScore_i^2 - 2 * predictedScore_i + 1` if trueLabel_i = 1 and
        // `predictedScore_i < 1` or `x_i = predictedScore^2` if `trueLabel_i = 0` and `predictedScore_i > 0`
        // or `x_i = 0` otherwise. The of those values is used as a denominator when calculating the gradients and
        // Hessians afterwards...
        float64 denominator = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;
            float64 x;

            if (trueLabel) {
                if (predictedScore < 1) {
                    x = (predictedScore * predictedScore) - (2 * predictedScore) + 1;
                } else {
                    x = 0;
                }
            } else {
                if (predictedScore > 0) {
                    x = (predictedScore * predictedScore);
                } else {
                    x = 0;
                }
            }

            statisticIterator[i].gradient = x;  // Temporarily store `x` in the array of gradients
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
            Statistic<float64>& statistic = statisticIterator[i];
            float64 gradient;
            float64 hessian;

            if (trueLabel) {
                if (predictedScore < 1) {
                    gradient = util::divideOrZero<float64>(predictedScore - 1, denominatorGradient);
                    hessian = util::divideOrZero<float64>(denominator - statistic.gradient, denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            } else {
                if (predictedScore > 0) {
                    gradient = util::divideOrZero<float64>(predictedScore, denominatorGradient);
                    hessian = util::divideOrZero<float64>(denominator - statistic.gradient, denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            }

            statistic.gradient = gradient;
            statistic.hessian = hessian;
            labelIterator2++;
        }
    }

    template<typename LabelIterator>
    static inline void updateNonDecomposableStatisticsInternally(
      View<float64>::const_iterator scoreIterator, LabelIterator labelIterator,
      DenseNonDecomposableStatisticView<float64>::gradient_iterator gradientIterator,
      DenseNonDecomposableStatisticView<float64>::hessian_iterator hessianIterator, uint32 numLabels) {
        LabelIterator labelIterator2 = labelIterator;
        LabelIterator labelIterator3 = labelIterator;

        // For each label `i`, calculate `x_i = predictedScore_i^2 - 2 * predictedScore_i + 1` if trueLabel_i = 1 and
        // `predictedScore_i < 1` or `x_i = predictedScore^2` if `trueLabel_i = 0` and `predictedScore_i > 0`
        // or `x_i = 0` otherwise. The of those values is used as a denominator when calculating the gradients and
        // Hessians afterwards...
        float64 denominator = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;
            float64 x;

            if (trueLabel) {
                if (predictedScore < 1) {
                    x = (predictedScore * predictedScore) - (2 * predictedScore) + 1;
                } else {
                    x = 0;
                }
            } else {
                if (predictedScore > 0) {
                    x = (predictedScore * predictedScore);
                } else {
                    x = 0;
                }
            }

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
            float64 gradient;
            float64 hessian;

            if (trueLabel) {
                if (predictedScore < 1) {
                    gradient = util::divideOrZero<float64>(predictedScore - 1, denominatorGradient);
                    hessian = util::divideOrZero<float64>(denominator - gradientIterator[i], denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            } else {
                if (predictedScore > 0) {
                    gradient = util::divideOrZero<float64>(predictedScore, denominatorGradient);
                    hessian = util::divideOrZero<float64>(denominator - gradientIterator[i], denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            }

            LabelIterator labelIterator4 = labelIterator3;

            for (uint32 j = 0; j < i; j++) {
                float64 hessianTriangle;

                if (!isEqualToZero(gradient)) {
                    bool trueLabel2 = *labelIterator4;
                    float64 predictedScore2 = scoreIterator[j];
                    float64 numerator;

                    if (trueLabel2) {
                        if (predictedScore2 < 1) {
                            numerator = predictedScore2 - 1;
                        } else {
                            numerator = 0;
                        }
                    } else {
                        if (predictedScore2 > 0) {
                            numerator = predictedScore2;
                        } else {
                            numerator = 0;
                        }
                    }

                    if (trueLabel) {
                        numerator *= -(predictedScore - 1);
                    } else {
                        numerator *= -predictedScore;
                    }

                    hessianTriangle = util::divideOrZero<float64>(numerator, denominatorHessian);
                } else {
                    hessianTriangle = 0;
                }

                *hessianIterator = hessianTriangle;
                hessianIterator++;
                labelIterator4++;
            }

            gradientIterator[i] = gradient;
            *hessianIterator = hessian;
            hessianIterator++;
            labelIterator2++;
        }
    }

    template<typename LabelIterator>
    static inline float64 evaluateInternally(View<float64>::const_iterator scoreIterator, LabelIterator labelIterator,
                                             uint32 numLabels) {
        // The example-wise squared hinge loss calculates as `sqrt((L_1 + ...)` with
        // `L_i = max(1 - predictedScore_i, 0)^2` if `trueLabel_i = 1` or `L_i = max(predictedScore_i, 0)^2` if
        // `trueLabel_i = 0`.
        float64 sumOfSquares = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;

            if (trueLabel) {
                if (predictedScore < 1) {
                    sumOfSquares += ((1 - predictedScore) * (1 - predictedScore));
                }
            } else {
                if (predictedScore > 0) {
                    sumOfSquares += (predictedScore * predictedScore);
                }
            }

            labelIterator++;
        }

        return std::sqrt(sumOfSquares);
    }

    /**
     * An implementation of the type `INonDecomposableClassificationLoss` that implements a multivariate variant of the
     * squared hinge loss that is non-decomposable.
     */
    class NonDecomposableSquaredHingeLoss final : public INonDecomposableClassificationLoss {
        public:

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<float64>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<float64>>& statisticView) const override {
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex),
                                                       labelMatrix.values_cbegin(exampleIndex),
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<float64>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<float64>>& statisticView) const override {
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex),
                                                       labelMatrix.values_cbegin(exampleIndex),
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<float64>& scoreMatrix,
              CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<float64>>& statisticView) const override {
                auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                       labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<float64>& scoreMatrix,
              PartialIndexVector::const_iterator indicesBegin, PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<float64>>& statisticView) const override {
                auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                       labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<float64>& scoreMatrix,
              DenseNonDecomposableStatisticView<float64>& statisticView) const override {
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.gradients_begin(exampleIndex), statisticView.hessians_begin(exampleIndex),
                  labelMatrix.numCols);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<float64>& scoreMatrix,
              DenseNonDecomposableStatisticView<float64>& statisticView) const override {
                auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                       labelMatrix.indices_cend(exampleIndex));
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelIterator, statisticView.gradients_begin(exampleIndex),
                  statisticView.hessians_begin(exampleIndex), labelMatrix.numCols);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          labelMatrix.values_cbegin(exampleIndex), labelMatrix.numCols);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override {
                auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
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
                auto labelIterator = createBinarySparseForwardIterator(labelVector.cbegin(), labelVector.cend());
                return evaluateInternally(scoresBegin, labelIterator, numLabels);
            }
    };

    /**
     * Allows to create instances of the type `INonDecomposableClassificationLoss` that implement a multivariate variant
     * of the squared hinge loss that is non-decomposable.
     */
    class NonDecomposableSquaredHingeLossFactory final : public INonDecomposableClassificationLossFactory {
        public:

            std::unique_ptr<INonDecomposableClassificationLoss> createNonDecomposableClassificationLoss()
              const override {
                return std::make_unique<NonDecomposableSquaredHingeLoss>();
            }

            std::unique_ptr<IDistanceMeasure> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const override {
                return this->createNonDecomposableClassificationLoss();
            }

            std::unique_ptr<IClassificationEvaluationMeasure> createClassificationEvaluationMeasure() const override {
                return this->createNonDecomposableClassificationLoss();
            }
    };

    NonDecomposableSquaredHingeLossConfig::NonDecomposableSquaredHingeLossConfig(
      ReadableProperty<IHeadConfig> headConfig)
        : headConfig_(headConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      NonDecomposableSquaredHingeLossConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return headConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, *this,
                                                                               blasFactory, lapackFactory);
    }

    std::unique_ptr<IMarginalProbabilityFunctionFactory>
      NonDecomposableSquaredHingeLossConfig::createMarginalProbabilityFunctionFactory() const {
        return nullptr;
    }

    std::unique_ptr<IJointProbabilityFunctionFactory>
      NonDecomposableSquaredHingeLossConfig::createJointProbabilityFunctionFactory() const {
        return nullptr;
    }

    float64 NonDecomposableSquaredHingeLossConfig::getDefaultPrediction() const {
        return 0.5;
    }

    std::unique_ptr<INonDecomposableClassificationLossFactory>
      NonDecomposableSquaredHingeLossConfig::createNonDecomposableClassificationLossFactory() const {
        return std::make_unique<NonDecomposableSquaredHingeLossFactory>();
    }

}
