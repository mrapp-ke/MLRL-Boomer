#include "boosting/losses/loss_example_wise_logistic.hpp"
#include "boosting/math/math.hpp"


namespace boosting {

    template<class LabelMatrix>
    static inline void updateExampleWiseStatisticsInternally(uint32 exampleIndex, const LabelMatrix& labelMatrix,
                                                             const CContiguousConstView<float64>& scoreMatrix,
                                                             DenseExampleWiseStatisticMatrix& statisticMatrix) {
        // This implementation uses the so-called "exp-normalize-trick" to increase numerical stability (see, e.g.,
        // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/). It is based on rewriting a fraction
        // of the form `exp(x_1) / (exp(x_1) + exp(x_2) + ...)` as
        // `exp(x_1 - max) / (exp(x_1 - max) + exp(x_2 - max) + ...)`, where `max = max(x_1, x_2, ...)`. To be able to
        // exploit this equivalence for the calculation of gradients and Hessians, they are calculated as products of
        // fractions of the above form.
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        typename LabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        DenseExampleWiseStatisticMatrix::gradient_iterator gradientIterator =
            statisticMatrix.gradients_row_begin(exampleIndex);
        DenseExampleWiseStatisticMatrix::hessian_iterator hessianIterator =
            statisticMatrix.hessians_row_begin(exampleIndex);
        uint32 numLabels = labelMatrix.getNumCols();

        // For each label `c`, calculate `x = -expectedScore_c * predictedScore_c` and find the largest and second
        // largest values (that must be greater than 0, because `exp(1) = 0`) among all of them...
        float64 max = 0;  // The largest value
        float64 max2 = 0;  // The second largest value

        for (uint32 c = 0; c < numLabels; c++) {
            float64 predictedScore = scoreIterator[c];
            uint32 trueLabel = *labelIterator;
            float64 x = trueLabel ? -predictedScore : predictedScore;
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
        float64 sumExp = std::exp(0.0 - max);
        float64 zeroExp = std::exp(0.0 - max2);
        float64 sumExp2 = zeroExp;

        for (uint32 c = 0; c < numLabels; c++) {
            float64 x = gradientIterator[c];
            sumExp += std::exp(x - max);
            sumExp2 += std::exp(x - max2);
        }

        // Calculate `zeroExp / sumExp2` (it is needed multiple times for calculating Hessians that belong to the upper
        // triangle of the Hessian matrix)...
        zeroExp = divideOrZero<float64>(zeroExp, sumExp2);

        // Calculate the gradients and Hessians...
        labelIterator = labelMatrix.row_values_cbegin(exampleIndex);

        for (uint32 c = 0; c < numLabels; c++) {
            float64 predictedScore = scoreIterator[c];
            uint8 trueLabel = *labelIterator;
            float64 invertedExpectedScore = trueLabel ? -1 : 1;
            float64 x = predictedScore * invertedExpectedScore;

            // Calculate the gradient that corresponds to the current label. The gradient calculates as
            // `-expectedScore_c * exp(x_c) / (1 + exp(x_1) + exp(x_2) + ...)`, which can be rewritten as
            // `-expectedScore_c * (exp(x_c - max) / sumExp)`
            float64 xExp = std::exp(x - max);
            float64 tmp = divideOrZero<float64>(xExp, sumExp);
            gradientIterator[c] = invertedExpectedScore * tmp;

            // Calculate the Hessians that belong to the part of the Hessian matrix' upper triangle that corresponds to
            // the current label. Such Hessian calculates as
            // `-expectedScore_c * expectedScore_r * exp(x_c + x_r) / (1 + exp(x_1) + exp(x_2) + ...)^2`, or as
            // `-expectedScore_c * expectedScore_r * (exp(x_c + x_r - max) / sumExp) * (exp(0 - max) / sumExp)`
            typename LabelMatrix::value_const_iterator labelIterator2 = labelMatrix.row_values_cbegin(exampleIndex);

            for (uint32 r = 0; r < c; r++) {
                float64 predictedScore2 = scoreIterator[r];
                uint32 trueLabel2 = *labelIterator2;
                float64 expectedScore2 = trueLabel2 ? 1 : -1;
                float64 x2 = predictedScore2 * -expectedScore2;
                *hessianIterator = invertedExpectedScore * expectedScore2
                                   * divideOrZero<float64>(std::exp(x + x2 - max2), sumExp2) * zeroExp;
                hessianIterator++;
                labelIterator2++;
            }

            // Calculate the Hessian on the diagonal of the Hessian matrix that corresponds to the current label. Such
            // Hessian calculates as `exp(x_c) * (1 + exp(x_1) + exp(x_2) + ...) / (1 + exp(x_1) + exp(x_2) + ...)^2`,
            // or as `(exp(x_c - max) / sumExp) * (1 - exp(x_c - max) / sumExp)`
            *hessianIterator = tmp * (1 - tmp);
            hessianIterator++;
            labelIterator++;
        }
    }

    template<class LabelMatrix>
    static inline float64 evaluateInternally(uint32 exampleIndex, const LabelMatrix& labelMatrix,
                                             const CContiguousConstView<float64>& scoreMatrix) {
        // The example-wise logistic loss calculates as
        // `log(1 + exp(-expectedScore_1 * predictedScore_1) + ... + exp(-expectedScore_2 * predictedScore_2) + ...)`.
        // In the following, we exploit the identity
        // `log(exp(x_1) + exp(x_2) + ...) = max + log(exp(x_1 - max) + exp(x_2 - max) + ...)`, where
        // `max = max(x_1, x_2, ...)`, to increase numerical stability (see, e.g., section "Log-sum-exp for computing
        // the log-distribution" in https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
        uint32 numLabels = labelMatrix.getNumCols();
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        typename LabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        float64 max = 0;

        // For each label `i`, calculate `x = -expectedScore_i * predictedScore_i` and find the largest value (that must
        // be greater than 0, because `exp(1) = 0`) among all of them...
        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = *labelIterator;
            float64 predictedScore = scoreIterator[i];
            float64 x = trueLabel ? -predictedScore : predictedScore;

            if (x > max) {
                max = x;
            }

            labelIterator++;
        }

        // Calculate the example-wise loss as `max + log(exp(0 - max) + exp(x_1 - max) + ...)`...
        float64 sumExp = std::exp(0 - max);
        labelIterator = labelMatrix.row_values_cbegin(exampleIndex);

        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = *labelIterator;
            float64 predictedScore = scoreIterator[i];
            float64 x = trueLabel ? -predictedScore : predictedScore;
            sumExp += std::exp(x - max);
            labelIterator++;
        }

        return max + std::log(sumExp);
    }

    void ExampleWiseLogisticLoss::updateExampleWiseStatistics(uint32 exampleIndex,
                                                              const CContiguousLabelMatrix& labelMatrix,
                                                              const CContiguousConstView<float64>& scoreMatrix,
                                                              DenseExampleWiseStatisticMatrix& statisticMatrix) const {
        updateExampleWiseStatisticsInternally<CContiguousLabelMatrix>(exampleIndex, labelMatrix, scoreMatrix,
                                                                      statisticMatrix);
    }

    void ExampleWiseLogisticLoss::updateExampleWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                              const CContiguousConstView<float64>& scoreMatrix,
                                                              DenseExampleWiseStatisticMatrix& statisticMatrix) const {
        updateExampleWiseStatisticsInternally<CsrLabelMatrix>(exampleIndex, labelMatrix, scoreMatrix, statisticMatrix);
    }

    float64 ExampleWiseLogisticLoss::evaluate(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                              const CContiguousConstView<float64>& scoreMatrix) const {
        return evaluateInternally<CContiguousLabelMatrix>(exampleIndex, labelMatrix, scoreMatrix);
    }

    float64 ExampleWiseLogisticLoss::evaluate(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                              const CContiguousConstView<float64>& scoreMatrix) const {
        return evaluateInternally<CsrLabelMatrix>(exampleIndex, labelMatrix, scoreMatrix);
    }

    float64 ExampleWiseLogisticLoss::measureSimilarity(const LabelVector& labelVector,
                                                       CContiguousConstView<float64>::const_iterator scoresBegin,
                                                       CContiguousConstView<float64>::const_iterator scoresEnd) const {
        // The example-wise logistic loss calculates as
        // `log(1 + exp(-expectedScore_1 * predictedScore_1) + ... + exp(-expectedScore_2 * predictedScore_2) + ...)`.
        // In the following, we exploit the identity
        // `log(exp(x_1) + exp(x_2) + ...) = max + log(exp(x_1 - max) + exp(x_2 - max) + ...)`, where
        // `max = max(x_1, x_2, ...)`, to increase numerical stability (see, e.g., section "Log-sum-exp for computing
        // the log-distribution" in https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
        uint32 numLabels = scoresEnd - scoresBegin;
        LabelVector::value_const_iterator labelIterator = labelVector.values_cbegin();
        float64 max = 0;

        // For each label `i`, calculate `x = -expectedScore_i * predictedScore_i` and find the largest value (that must
        // be greater than 0, because `exp(1) = 0`) among all of them...
        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoresBegin[i];
            bool trueLabel = *labelIterator;
            float64 x = trueLabel ? -predictedScore : predictedScore;

            if (x > max) {
                max = x;
            }

            labelIterator++;
        }

        // Calculate the example-wise loss as `max + log(exp(0 - max) + exp(x_1 - max) + ...)`...
        float64 sumExp = std::exp(0 - max);
        labelIterator = labelVector.values_cbegin();

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoresBegin[i];
            bool trueLabel = *labelIterator;
            float64 x = trueLabel ? -predictedScore : predictedScore;
            sumExp += std::exp(x - max);
            labelIterator++;
        }

        return max + std::log(sumExp);
    }

}