#include "mlrl/boosting/losses/loss_non_decomposable_logistic.hpp"

#include "mlrl/boosting/prediction/probability_function_chain_rule.hpp"
#include "mlrl/boosting/prediction/probability_function_logistic.hpp"
#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/util/math.hpp"

namespace boosting {

    template<typename StatisticType>
    static inline void updateGradientAndHessian(StatisticType invertedExpectedScore, StatisticType x, StatisticType max,
                                                StatisticType sumExp, StatisticType& gradient, StatisticType& hessian) {
        // Calculate the gradient that corresponds to the current label. The gradient calculates as
        // `-expectedScore_c * exp(x_c) / (1 + exp(x_1) + exp(x_2) + ...)`, which can be rewritten as
        // `-expectedScore_c * (exp(x_c - max) / sumExp)`
        StatisticType xExp = std::exp(x - max);
        StatisticType tmp = util::divideOrZero(xExp, sumExp);
        gradient = invertedExpectedScore * tmp;

        // Calculate the Hessian on the diagonal of the Hessian matrix that corresponds to the current label. Such
        // Hessian calculates as `exp(x_c) * (1 + exp(x_1) + exp(x_2) + ...) / (1 + exp(x_1) + exp(x_2) + ...)^2`,
        // or as `(exp(x_c - max) / sumExp) * (1 - exp(x_c - max) / sumExp)`
        hessian = tmp * (1 - tmp);
    }

    template<typename ScoreIterator, typename LabelIterator, typename StatisticIterator>
    static inline void updateDecomposableStatisticsInternally(ScoreIterator scoreIterator, LabelIterator labelIterator,
                                                              StatisticIterator statisticIterator, uint32 numLabels) {
        typedef typename util::iterator_value<ScoreIterator> statistic_type;

        // This implementation uses the so-called "exp-normalize-trick" to increase numerical stability (see, e.g.,
        // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/). It is based on rewriting a fraction
        // of the form `exp(x_1) / (exp(x_1) + exp(x_2) + ...)` as
        // `exp(x_1 - max) / (exp(x_1 - max) + exp(x_2 - max) + ...)`, where `max = max(x_1, x_2, ...)`. To be able to
        // exploit this equivalence for the calculation of gradients and Hessians, they are calculated as products of
        // fractions of the above form.
        LabelIterator labelIterator2 = labelIterator;

        // For each label `c`, calculate `x = -expectedScore_c * predictedScore_c` and find the largest and second
        // largest values (that must be greater than 0, because `exp(1) = 0`) among all of them...
        statistic_type max = 0;  // The largest value

        for (uint32 c = 0; c < numLabels; c++) {
            statistic_type predictedScore = scoreIterator[c];
            bool trueLabel = *labelIterator;
            statistic_type x = trueLabel ? -predictedScore : predictedScore;
            statisticIterator[c].gradient = x;  // Temporarily store `x` in the array of statistics

            if (x > max) {
                max = x;
            }

            labelIterator++;
        }

        // Calculate `sumExp = exp(0 - max) + exp(x_1 - max) + exp(x_2 - max) + ...`
        statistic_type sumExp = std::exp(0.0 - max);

        for (uint32 c = 0; c < numLabels; c++) {
            statistic_type x = statisticIterator[c].gradient;
            sumExp += std::exp(x - max);
        }

        // Calculate the gradients and Hessians...
        for (uint32 c = 0; c < numLabels; c++) {
            statistic_type predictedScore = scoreIterator[c];
            bool trueLabel = *labelIterator2;
            statistic_type invertedExpectedScore = trueLabel ? -1 : 1;
            statistic_type x = predictedScore * invertedExpectedScore;
            Statistic<statistic_type>& statistic = statisticIterator[c];
            updateGradientAndHessian(invertedExpectedScore, x, max, sumExp, statistic.gradient, statistic.hessian);
            labelIterator2++;
        }
    }

    template<typename ScoreIterator, typename LabelIterator, typename GradientIterator, typename HessianIterator>
    static inline void updateNonDecomposableStatisticsInternally(ScoreIterator scoreIterator,
                                                                 LabelIterator labelIterator,
                                                                 GradientIterator gradientIterator,
                                                                 HessianIterator hessianIterator, uint32 numLabels) {
        typedef typename util::iterator_value<GradientIterator> statistic_type;

        // This implementation uses the so-called "exp-normalize-trick" to increase numerical stability (see, e.g.,
        // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/). It is based on rewriting a fraction
        // of the form `exp(x_1) / (exp(x_1) + exp(x_2) + ...)` as
        // `exp(x_1 - max) / (exp(x_1 - max) + exp(x_2 - max) + ...)`, where `max = max(x_1, x_2, ...)`. To be able to
        // exploit this equivalence for the calculation of gradients and Hessians, they are calculated as products of
        // fractions of the above form.
        LabelIterator labelIterator2 = labelIterator;
        LabelIterator labelIterator3 = labelIterator;

        // For each label `c`, calculate `x = -expectedScore_c * predictedScore_c` and find the largest and second
        // largest values (that must be greater than 0, because `exp(1) = 0`) among all of them...
        statistic_type max = 0;   // The largest value
        statistic_type max2 = 0;  // The second largest value

        for (uint32 c = 0; c < numLabels; c++) {
            statistic_type predictedScore = scoreIterator[c];
            bool trueLabel = *labelIterator;
            statistic_type x = trueLabel ? -predictedScore : predictedScore;
            gradientIterator[c] = x;  // Temporarily store `x` in the array of gradients

            if (x > max) {
                max2 = max;
                max = x;
            } else if (x > max2) {
                max2 = x;
            }

            labelIterator++;
        }

        // In the following, the largest value the exponential function may be applied to is `max + max2`, which happens
        // when Hessians that belong to the upper triangle of the Hessian matrix are calculated...
        max2 += max;

        // Calculate `sumExp = exp(0 - max) + exp(x_1 - max) + exp(x_2 - max) + ...`
        statistic_type sumExp = std::exp(0.0 - max);
        statistic_type zeroExp = std::exp(0.0 - max2);
        statistic_type sumExp2 = zeroExp;

        for (uint32 c = 0; c < numLabels; c++) {
            statistic_type x = gradientIterator[c];
            sumExp += std::exp(x - max);
            sumExp2 += std::exp(x - max2);
        }

        // Calculate `zeroExp / sumExp2` (it is needed multiple times for calculating Hessians that belong to the upper
        // triangle of the Hessian matrix)...
        zeroExp = util::divideOrZero(zeroExp, sumExp2);

        // Calculate the gradients and Hessians...
        for (uint32 c = 0; c < numLabels; c++) {
            statistic_type predictedScore = scoreIterator[c];
            bool trueLabel = *labelIterator2;
            statistic_type invertedExpectedScore = trueLabel ? -1 : 1;
            statistic_type x = predictedScore * invertedExpectedScore;

            // Calculate the Hessians that belong to the part of the Hessian matrix' upper triangle that corresponds to
            // the current label. Such Hessian calculates as
            // `-expectedScore_c * expectedScore_r * exp(x_c + x_r) / (1 + exp(x_1) + exp(x_2) + ...)^2`, or as
            // `-expectedScore_c * expectedScore_r * (exp(x_c + x_r - max) / sumExp) * (exp(0 - max) / sumExp)`
            LabelIterator labelIterator4 = labelIterator3;

            for (uint32 r = 0; r < c; r++) {
                statistic_type predictedScore2 = scoreIterator[r];
                bool trueLabel2 = *labelIterator4;
                statistic_type expectedScore2 = trueLabel2 ? 1 : -1;
                statistic_type x2 = predictedScore2 * -expectedScore2;
                *hessianIterator = invertedExpectedScore * expectedScore2
                                   * util::divideOrZero(std::exp(x + x2 - max2), sumExp2) * zeroExp;
                hessianIterator++;
                labelIterator4++;
            }

            updateGradientAndHessian(invertedExpectedScore, x, max, sumExp, gradientIterator[c], *hessianIterator);
            hessianIterator++;
            labelIterator2++;
        }
    }

    template<typename ScoreIterator, typename LabelIterator>
    static inline typename util::iterator_value<ScoreIterator> evaluateInternally(ScoreIterator scoreIterator,
                                                                                  LabelIterator labelIterator,
                                                                                  uint32 numLabels) {
        typedef typename util::iterator_value<ScoreIterator> score_type;

        // The example-wise logistic loss calculates as
        // `log(1 + exp(-expectedScore_1 * predictedScore_1) + ... + exp(-expectedScore_2 * predictedScore_2) + ...)`.
        // In the following, we exploit the identity
        // `log(exp(x_1) + exp(x_2) + ...) = max + log(exp(x_1 - max) + exp(x_2 - max) + ...)`, where
        // `max = max(x_1, x_2, ...)`, to increase numerical stability (see, e.g., section "Log-sum-exp for computing
        // the log-distribution" in https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
        LabelIterator labelIterator2 = labelIterator;
        score_type max = 0;

        // For each label `i`, calculate `x = -expectedScore_i * predictedScore_i` and find the largest value (that must
        // be greater than 0, because `exp(1) = 0`) among all of them...
        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = *labelIterator;
            score_type predictedScore = scoreIterator[i];
            score_type x = trueLabel ? -predictedScore : predictedScore;

            if (x > max) {
                max = x;
            }

            labelIterator++;
        }

        // Calculate the example-wise loss as `max + log(exp(0 - max) + exp(x_1 - max) + ...)`...
        score_type sumExp = std::exp(0 - max);

        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = *labelIterator2;
            score_type predictedScore = scoreIterator[i];
            score_type x = trueLabel ? -predictedScore : predictedScore;
            sumExp += std::exp(x - max);
            labelIterator2++;
        }

        return max + std::log(sumExp);
    }

    /**
     * An implementation of the type `INonDecomposableClassificationLoss` that implements a multivariate variant of the
     * logistic loss that is non-decomposable.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class NonDecomposableLogisticLoss final : public INonDecomposableClassificationLoss<StatisticType> {
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
                auto labelIterator = createBinarySparseForwardIterator(labelVector.cbegin(), labelVector.cend());
                uint32 numLabels = scoresEnd - scoresBegin;
                return evaluateInternally(scoresBegin, labelIterator, numLabels);
            }
    };

    /**
     * Allows to create instances of the type `INonDecomposableClassificationLoss` that implement a multivariate variant
     * of the logistic loss that is non-decomposable.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class NonDecomposableLogisticLossFactory final : public INonDecomposableClassificationLossFactory<StatisticType> {
        public:

            std::unique_ptr<INonDecomposableClassificationLoss<StatisticType>> createNonDecomposableClassificationLoss()
              const override {
                return std::make_unique<NonDecomposableLogisticLoss<StatisticType>>();
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

    NonDecomposableLogisticLossConfig::NonDecomposableLogisticLossConfig(ReadableProperty<IHeadConfig> headConfig)
        : headConfig_(headConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      NonDecomposableLogisticLossConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory, bool preferSparseStatistics) const {
        return headConfig_.get().createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix, *this,
                                                                               blasFactory, lapackFactory);
    }

    std::unique_ptr<IMarginalProbabilityFunctionFactory>
      NonDecomposableLogisticLossConfig::createMarginalProbabilityFunctionFactory() const {
        return std::make_unique<LogisticFunctionFactory>();
    }

    std::unique_ptr<IJointProbabilityFunctionFactory>
      NonDecomposableLogisticLossConfig::createJointProbabilityFunctionFactory() const {
        return std::make_unique<ChainRuleFactory>(this->createMarginalProbabilityFunctionFactory());
    }

    float64 NonDecomposableLogisticLossConfig::getDefaultPrediction() const {
        return 0;
    }

    std::unique_ptr<INonDecomposableClassificationLossFactory<float64>>
      NonDecomposableLogisticLossConfig::createNonDecomposableClassificationLossFactory() const {
        return std::make_unique<NonDecomposableLogisticLossFactory<float64>>();
    }

}
