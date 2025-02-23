#include "mlrl/boosting/losses/loss_non_decomposable_squared_hinge.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/util/math.hpp"

namespace boosting {

    template<typename ScoreIterator, typename LabelIterator, typename StatisticIterator>
    static inline void updateDecomposableStatisticsInternally(ScoreIterator scoreIterator, LabelIterator labelIterator,
                                                              StatisticIterator statisticIterator, uint32 numLabels) {
        typedef typename util::iterator_value<ScoreIterator> statistic_type;
        LabelIterator labelIterator2 = labelIterator;

        // For each label `i`, calculate `x_i = predictedScore_i^2 - 2 * predictedScore_i + 1` if trueLabel_i = 1 and
        // `predictedScore_i < 1` or `x_i = predictedScore^2` if `trueLabel_i = 0` and `predictedScore_i > 0`
        // or `x_i = 0` otherwise. The of those values is used as a denominator when calculating the gradients and
        // Hessians afterwards...
        statistic_type denominator = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            statistic_type predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;
            statistic_type x;

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
        statistic_type denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        statistic_type denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numLabels; i++) {
            statistic_type predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator2;
            Statistic<statistic_type>& statistic = statisticIterator[i];
            statistic_type gradient;
            statistic_type hessian;

            if (trueLabel) {
                if (predictedScore < 1) {
                    gradient = util::divideOrZero(predictedScore - 1, denominatorGradient);
                    hessian = util::divideOrZero(denominator - statistic.gradient, denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            } else {
                if (predictedScore > 0) {
                    gradient = util::divideOrZero(predictedScore, denominatorGradient);
                    hessian = util::divideOrZero(denominator - statistic.gradient, denominatorHessian);
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

    template<typename ScoreIterator, typename LabelIterator, typename GradientIterator, typename HessianIterator>
    static inline void updateNonDecomposableStatisticsInternally(ScoreIterator scoreIterator,
                                                                 LabelIterator labelIterator,
                                                                 GradientIterator gradientIterator,
                                                                 HessianIterator hessianIterator, uint32 numLabels) {
        typedef typename util::iterator_value<ScoreIterator> statistic_type;
        LabelIterator labelIterator2 = labelIterator;
        LabelIterator labelIterator3 = labelIterator;

        // For each label `i`, calculate `x_i = predictedScore_i^2 - 2 * predictedScore_i + 1` if trueLabel_i = 1 and
        // `predictedScore_i < 1` or `x_i = predictedScore^2` if `trueLabel_i = 0` and `predictedScore_i > 0`
        // or `x_i = 0` otherwise. The of those values is used as a denominator when calculating the gradients and
        // Hessians afterwards...
        statistic_type denominator = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            statistic_type predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;
            statistic_type x;

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
        statistic_type denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        statistic_type denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numLabels; i++) {
            statistic_type predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator2;
            statistic_type gradient;
            statistic_type hessian;

            if (trueLabel) {
                if (predictedScore < 1) {
                    gradient = util::divideOrZero(predictedScore - 1, denominatorGradient);
                    hessian = util::divideOrZero(denominator - gradientIterator[i], denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            } else {
                if (predictedScore > 0) {
                    gradient = util::divideOrZero(predictedScore, denominatorGradient);
                    hessian = util::divideOrZero(denominator - gradientIterator[i], denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            }

            LabelIterator labelIterator4 = labelIterator3;

            for (uint32 j = 0; j < i; j++) {
                statistic_type hessianTriangle;

                if (!isEqualToZero(gradient)) {
                    bool trueLabel2 = *labelIterator4;
                    statistic_type predictedScore2 = scoreIterator[j];
                    statistic_type numerator;

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

                    hessianTriangle = util::divideOrZero(numerator, denominatorHessian);
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

    template<typename ScoreIterator, typename LabelIterator>
    static inline typename util::iterator_value<ScoreIterator> evaluateInternally(ScoreIterator scoreIterator,
                                                                                  LabelIterator labelIterator,
                                                                                  uint32 numLabels) {
        typedef typename util::iterator_value<ScoreIterator> score_type;

        // The example-wise squared hinge loss calculates as `sqrt((L_1 + ...)` with
        // `L_i = max(1 - predictedScore_i, 0)^2` if `trueLabel_i = 1` or `L_i = max(predictedScore_i, 0)^2` if
        // `trueLabel_i = 0`.
        score_type sumOfSquares = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            score_type predictedScore = scoreIterator[i];
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
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class NonDecomposableSquaredHingeLoss final : public INonDecomposableClassificationLoss<StatisticType> {
        public:

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex),
                                                       labelMatrix.values_cbegin(exampleIndex),
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex),
                                                       labelMatrix.values_cbegin(exampleIndex),
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                       labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            virtual void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              PartialIndexVector::const_iterator indicesBegin, PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override {
                auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                       labelMatrix.indices_cend(exampleIndex));
                updateDecomposableStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator,
                                                       statisticView.values_begin(exampleIndex), labelMatrix.numCols);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const override {
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.gradients_begin(exampleIndex), statisticView.hessians_begin(exampleIndex),
                  labelMatrix.numCols);
            }

            void updateNonDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              DenseNonDecomposableStatisticView<StatisticType>& statisticView) const override {
                auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                       labelMatrix.indices_cend(exampleIndex));
                updateNonDecomposableStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelIterator, statisticView.gradients_begin(exampleIndex),
                  statisticView.hessians_begin(exampleIndex), labelMatrix.numCols);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          labelMatrix.values_cbegin(exampleIndex), labelMatrix.numCols);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override {
                auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                       labelMatrix.indices_cend(exampleIndex));
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator, labelMatrix.numCols);
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            StatisticType measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                          typename View<StatisticType>::const_iterator scoresBegin,
                                          typename View<StatisticType>::const_iterator scoresEnd) const override {
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = createBinarySparseForwardIterator(labelVector.cbegin(), labelVector.cend());
                return evaluateInternally(scoresBegin, labelIterator, numLabels);
            }
    };

    /**
     * Allows to create instances of the type `INonDecomposableClassificationLoss` that implement a multivariate variant
     * of the squared hinge loss that is non-decomposable.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class NonDecomposableSquaredHingeLossFactory final
        : public INonDecomposableClassificationLossFactory<StatisticType> {
        public:

            std::unique_ptr<INonDecomposableClassificationLoss<StatisticType>> createNonDecomposableClassificationLoss()
              const override {
                return std::make_unique<NonDecomposableSquaredHingeLoss<StatisticType>>();
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
    };

    NonDecomposableSquaredHingeLossConfig::NonDecomposableSquaredHingeLossConfig(
      ReadableProperty<IStatisticTypeConfig> statisticTypeConfig)
        : statisticTypeConfig_(statisticTypeConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      NonDecomposableSquaredHingeLossConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return statisticTypeConfig_.get().createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, *this, blasFactory, lapackFactory);
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

    std::unique_ptr<INonDecomposableClassificationLossFactory<float64>>
      NonDecomposableSquaredHingeLossConfig::createNonDecomposableClassificationLossFactory() const {
        return std::make_unique<NonDecomposableSquaredHingeLossFactory<float64>>();
    }

}
