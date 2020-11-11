#include "losses_example_wise.h"
#include <cmath>

using namespace boosting;


void ExampleWiseLogisticLossImpl::updateStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                                   const DenseNumericMatrix<float64>& predictedScores,
                                                   DenseExampleWiseStatisticsMatrix& statistics) const {
    DenseExampleWiseStatisticsMatrix::gradient_iterator gradientIterator = statistics.gradients_row_begin(exampleIndex);
    DenseExampleWiseStatisticsMatrix::hessian_iterator hessianIterator = statistics.hessians_row_begin(exampleIndex);
    DenseNumericMatrix<float64>::const_iterator scoreIterator = predictedScores.row_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.getNumLabels();
    float64 sumOfExponentials = 1;

    for (uint32 c = 0; c < numLabels; c++) {
        uint8 trueLabel = labelMatrix.getValue(exampleIndex, c);
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 predictedScore = scoreIterator[c];
        float64 exponential = exp(-expectedScore * predictedScore);
        gradientIterator[c] = exponential;  // Temporarily store the exponential in the existing output array
        sumOfExponentials += exponential;
    }

    float64 sumOfExponentialsPow = sumOfExponentials * sumOfExponentials;
    uint32 i = 0;

    for (uint32 c = 0; c < numLabels; c++) {
        uint8 trueLabel = labelMatrix.getValue(exampleIndex, c);
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 predictedScore = scoreIterator[c];
        float64 exponential = gradientIterator[c];
        float64 tmp = (-expectedScore * exponential) / sumOfExponentials;
        gradientIterator[c] = tmp;

        for (uint32 c2 = 0; c2 < c; c2++) {
            trueLabel = labelMatrix.getValue(exampleIndex, c2);
            float64 expectedScore2 = trueLabel ? 1 : -1;
            float64 predictedScore2 = scoreIterator[c2];
            tmp = exp((-expectedScore2 * predictedScore2) - (expectedScore * predictedScore));
            tmp *= -expectedScore2 * expectedScore;
            tmp /= sumOfExponentialsPow;
            hessianIterator[i] = tmp;
            i++;
        }

        tmp = expectedScore * expectedScore * exponential * (sumOfExponentials - exponential);
        tmp /= sumOfExponentialsPow;
        hessianIterator[i] = tmp;
        i++;
    }
}
