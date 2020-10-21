#include "label_wise_losses.h"
#include <cmath>

using namespace boosting;


std::pair<float64, float64> LabelWiseLogisticLossImpl::calculateGradientAndHessian(
        const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex, uint32 labelIndex,
        float64 predictedScore) const {
    uint8 trueLabel = labelMatrix.getValue(exampleIndex, labelIndex);
    float64 expectedScore = trueLabel ? 1 : -1;
    float64 exponential = exp(expectedScore * predictedScore);
    float64 gradient = -expectedScore / (1 + exponential);
    float64 hessian = (pow(expectedScore, 2) * exponential) / pow(1 + exponential, 2);
    return std::make_pair(gradient, hessian);
}

std::pair<float64, float64> LabelWiseSquaredErrorLossImpl::calculateGradientAndHessian(
        const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex, uint32 labelIndex,
        float64 predictedScore) const {
    uint8 trueLabel = labelMatrix.getValue(exampleIndex, labelIndex);
    float64 expectedScore = trueLabel ? 1 : -1;
    float64 gradient = (2 * predictedScore) - (2 * expectedScore);
    float64 hessian = 2;
    return std::make_pair(gradient, hessian);
}

std::pair<float64, float64> LabelWiseSquaredHingeLossImpl::calculateGradientAndHessian(
        const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex, uint32 labelIndex,
        float64 predictedScore) const {
    uint8 trueLabel = labelMatrix.getValue(exampleIndex, labelIndex);
    float64 gradient;
    float64 hessian;

    if (trueLabel) {
        if (predictedScore < 1) {
            gradient = 2 * (predictedScore - 1);
            hessian = 2;
        } else {
            gradient = 0;
            hessian = 0;
        }
    } else {
        if (predictedScore > 0) {
            gradient = 2 * predictedScore;
            hessian = 2;
        } else {
            gradient = 0;
            hessian = 0;
        }
    }

    return std::make_pair(gradient, hessian);
}
