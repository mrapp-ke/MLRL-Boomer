#include "loss_example_wise_logistic.h"
#include "../math/math.h"

using namespace boosting;


void ExampleWiseLogisticLoss::updateExampleWiseStatistics(uint32 exampleIndex,
                                                          const IRandomAccessLabelMatrix& labelMatrix,
                                                          const DenseNumericMatrix<float64>& scoreMatrix,
                                                          DenseExampleWiseStatisticMatrix& statisticMatrix) const {
    // This implementation uses the so-called "exp-normalize-trick" to increase numerical stability (see, e.g.,
    // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/). It is based on rewritten a fraction of
    // the form `exp(x_1) / (exp(x_1) + exp(x_2) + ...)` as `exp(x_1 - max) / (exp(x_1 - max) + exp(x_2 - max) + ...)`,
    // where `max = max(x_1, x_2, ...)`. To be able to exploit this equivalence for the calculation of gradients and
    // Hessians, they are calculated as products of fractions of the above form.
    DenseNumericMatrix<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
    DenseExampleWiseStatisticMatrix::gradient_iterator gradientIterator =
        statisticMatrix.gradients_row_begin(exampleIndex);
    DenseExampleWiseStatisticMatrix::hessian_iterator hessianIterator =
        statisticMatrix.hessians_row_begin(exampleIndex);
    uint32 numLabels = labelMatrix.getNumCols();

    // For each label `c`, calculate `x = -expectedScore_c * predictedScore_c` and find the largest and second largest
    // values (that must be greater than 0, because `exp(1) = 0`) among all of them...
    float64 max = 0;  // The largest value
    float64 max2 = 0;  // The second largest value

    for (uint32 c = 0; c < numLabels; c++) {
        float64 predictedScore = scoreIterator[c];
        uint32 trueLabel = labelMatrix.getValue(exampleIndex, c);
        float64 x = trueLabel ? -predictedScore : predictedScore;
        gradientIterator[c] = x;  // Temporarily store `x` in the array of gradients

        if (x > max) {
            max2 = max;
            max = x;
        } else if (x > max2) {
            max2 = x;
        }
    }

    // Calculate `sumExp = exp(0 - max) + exp(x_1 - max) + exp(x_2 - max) + ...`
    float64 zeroExp = std::exp(0.0 - max);
    float64 sumExp = zeroExp;

    for (uint32 c = 0; c < numLabels; c++) {
        float64 x = gradientIterator[c];
        sumExp += std::exp(x - max);
    }

    // In the following, the largest value the exponential function may be applied to is `max + max2`, which happens
    // when Hessians that belong to the upper triangle of the Hessian matrix are calculated...
    max2 += max;
    float64 sumExp2;

    // If `max + max2 != max` we must calculate `sumExp2 = exp(0 - max2) + exp(x_1 - max2) + exp(x_2 - max2) + ...`,
    // otherwise it's the same as `sumExp`...
    if (max2 != max) {
        zeroExp = std::exp(0.0 - max2);
        sumExp2 = zeroExp;

        for (uint32 c = 0; c < numLabels; c++) {
            float64 x = gradientIterator[c];
            sumExp2 += std::exp(x - max2);
        }
    } else {
        sumExp2 = sumExp;
    }

    // Calculate `zeroExp / sumExp2` (it is needed multiple times for calculating Hessians that belong to the upper
    // triangle of the Hessian matrix)...
    zeroExp = divideOrZero<float64>(zeroExp, sumExp2);

    // Calculate the gradients and Hessians by traversing the labels in reverse order (to ensure that the values that
    // have temporarily been stored in the array of gradients have not been overwritten yet)
    intp i = triangularNumber(numLabels) - 1;

    for (intp c = numLabels - 1; c >= 0; c--) {
        uint8 trueLabel = labelMatrix.getValue(exampleIndex, c);
        float64 invertedExpectedScore = trueLabel ? -1 : 1;
        float64 x = gradientIterator[c];

        // Calculate the gradient that corresponds to the current label. The gradient calculates as
        // `-expectedScore_c * exp(x_c) / (1 + exp(x_1) + exp(x_2) + ...)`, which can be rewritten as
        // `-expectedScore_c * (exp(x_c - max) / sumExp)`
        float64 xExp = std::exp(x - max);
        float64 tmp = divideOrZero<float64>(xExp, sumExp);
        float64 gradient = invertedExpectedScore * tmp;

        // Calculate the Hessian on the diagonal of the Hessian matrix that corresponds to the current label. Such
        // Hessian calculates as `exp(x_c) * (1 + exp(x_1) + exp(x_2) + ...) / (1 + exp(x_1) + exp(x_2) + ...)^2`, or as
        // `(exp(x_c - max) / sumExp) * (1 - exp(x_c - max) / sumExp)`
        hessianIterator[i] = tmp * (1 - tmp);
        i--;

        // Calculate the Hessians that belong to the part of the Hessian matrix' upper triangle that corresponds to the
        // current label. Such Hessian calculates as
        // `-expectedScore_c * expectedScore_r * exp(x_c + x_r) / (1 + exp(x_1) + exp(x_2) + ...)^2`, or equivalently as
        // `-expectedScore_c * expectedScore_r * (exp(x_c + x_r - max) / sumExp) * (exp(0 - max) / sumExp)`
        for (intp r = c - 1; r >= 0; r--) {
            uint32 trueLabel2 = labelMatrix.getValue(exampleIndex, r);
            float64 expectedScore2 = trueLabel2 ? 1 : -1;
            float64 x2 = gradientIterator[r];
            hessianIterator[i] = invertedExpectedScore * expectedScore2
                                 * divideOrZero<float64>(std::exp(x + x2 - max2), sumExp2) * zeroExp;
            i--;
        }

        gradientIterator[c] = gradient;
    }
}
