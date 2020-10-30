#include "losses_example_wise.h"
#include <cmath>

using namespace boosting;


void ExampleWiseLogisticLossImpl::calculateGradientsAndHessians(const IRandomAccessLabelMatrix& labelMatrix,
                                                                uint32 exampleIndex, const float64* predictedScores,
                                                                float64* gradients, float64* hessians) const {
    uint32 numLabels = labelMatrix.getNumCols();
    float64 sumOfExponentials = 1;

    for (uint32 c = 0; c < numLabels; c++) {
        uint8 trueLabel = labelMatrix.getValue(exampleIndex, c);
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 predictedScore = predictedScores[c];
        float64 exponential = exp(-expectedScore * predictedScore);
        gradients[c] = exponential;  // Temporarily store the exponential in the existing output array
        sumOfExponentials += exponential;
    }

    float64 sumOfExponentialsPow = sumOfExponentials * sumOfExponentials;
    uint32 i = 0;

    for (uint32 c = 0; c < numLabels; c++) {
        uint8 trueLabel = labelMatrix.getValue(exampleIndex, c);
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 predictedScore = predictedScores[c];
        float64 exponential = gradients[c];
        float64 tmp = (-expectedScore * exponential) / sumOfExponentials;
        gradients[c] = tmp;

        for (uint32 c2 = 0; c2 < c; c2++) {
            trueLabel = labelMatrix.getValue(exampleIndex, c2);
            float64 expectedScore2 = trueLabel ? 1 : -1;
            float64 predictedScore2 = predictedScores[c2];
            tmp = exp((-expectedScore2 * predictedScore2) - (expectedScore * predictedScore));
            tmp *= -expectedScore2 * expectedScore;
            tmp /= sumOfExponentialsPow;
            hessians[i] = tmp;
            i++;
        }

        tmp = expectedScore * expectedScore * exponential * (sumOfExponentials - exponential);
        tmp /= sumOfExponentialsPow;
        hessians[i] = tmp;
        i++;
    }
}
