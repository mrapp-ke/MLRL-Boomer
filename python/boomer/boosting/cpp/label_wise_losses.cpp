#include "label_wise_losses.h"
#include <math.h>

using namespace boosting;


std::pair<float64, float64> LabelWiseLogisticLossImpl::calculateGradientAndHessian(
        IRandomAccessLabelMatrix* labelMatrix, uint32 exampleIndex, uint32 labelIndex, float64 predictedScore) {
    uint8 trueLabel = labelMatrix->getValue(exampleIndex, labelIndex);
    float64 expectedScore = trueLabel ? 1 : -1;
    float64 exponential = exp(expectedScore * predictedScore);
    float64 gradient = -expectedScore / (1 + exponential);
    float64 hessian = (pow(expectedScore, 2) * exponential) / pow(1 + exponential, 2);
    return std::make_pair(gradient, hessian);
}

std::pair<float64, float64> LabelWiseSquaredErrorLossImpl::calculateGradientAndHessian(
        IRandomAccessLabelMatrix* labelMatrix, uint32 exampleIndex, uint32 labelIndex, float64 predictedScore) {
    uint8 trueLabel = labelMatrix->getValue(exampleIndex, labelIndex);
    float64 expectedScore = trueLabel ? 1 : -1;
    float64 gradient = (2 * predictedScore) - (2 * expectedScore);
    float64 hessian = 2;
    return std::make_pair(gradient, hessian);
}
