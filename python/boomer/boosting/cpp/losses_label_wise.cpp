#include "losses_label_wise.h"
#include <cmath>

using namespace boosting;


void AbstractLabelWiseLoss::updateStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                             const DenseNumericMatrix<float64>& predictedScores,
                                             const FullIndexVector::const_iterator labelIndicesBegin,
                                             const FullIndexVector::const_iterator labelIndicesEnd,
                                             DenseLabelWiseStatisticMatrix& statistics) const {
    DenseLabelWiseStatisticMatrix::gradient_iterator gradientIterator = statistics.gradients_row_begin(exampleIndex);
    DenseLabelWiseStatisticMatrix::hessian_iterator hessianIterator = statistics.hessians_row_begin(exampleIndex);
    DenseNumericMatrix<float64>::const_iterator scoreIterator = predictedScores.row_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.getNumLabels();

    for (uint32 i = 0; i < numLabels; i++) {
        bool trueLabel = labelMatrix.getValue(exampleIndex, i);
        float64 predictedScore = scoreIterator[i];
        this->updateGradientAndHessian(&gradientIterator[i], &hessianIterator[i], trueLabel, predictedScore);
    }
}

void AbstractLabelWiseLoss::updateStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                             const DenseNumericMatrix<float64>& predictedScores,
                                             const PartialIndexVector::const_iterator labelIndicesBegin,
                                             const PartialIndexVector::const_iterator labelIndicesEnd,
                                             DenseLabelWiseStatisticMatrix& statistics) const {
    DenseLabelWiseStatisticMatrix::gradient_iterator gradientIterator = statistics.gradients_row_begin(exampleIndex);
    DenseLabelWiseStatisticMatrix::hessian_iterator hessianIterator = statistics.hessians_row_begin(exampleIndex);
    DenseNumericMatrix<float64>::const_iterator scoreIterator = predictedScores.row_cbegin(exampleIndex);

    for (auto indexIterator = labelIndicesBegin; indexIterator != labelIndicesEnd; indexIterator++) {
        uint32 labelIndex = *indexIterator;
        bool trueLabel = labelMatrix.getValue(exampleIndex, labelIndex);
        float64 predictedScore = scoreIterator[labelIndex];
        this->updateGradientAndHessian(&gradientIterator[labelIndex], &hessianIterator[labelIndex], trueLabel,
                                       predictedScore);
    }
}

void LabelWiseLogisticLossImpl::updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                                         DenseVector<float64>::iterator hessian, bool trueLabel,
                                                         float64 predictedScore) const {
    float64 expectedScore = trueLabel ? 1 : -1;
    float64 exponential = exp(expectedScore * predictedScore);
    *gradient = -expectedScore / (1 + exponential);
    *hessian = (pow(expectedScore, 2) * exponential) / pow(1 + exponential, 2);
}

void LabelWiseSquaredErrorLossImpl::updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                                             DenseVector<float64>::iterator hessian, bool trueLabel,
                                                             float64 predictedScore) const {
    float64 expectedScore = trueLabel ? 1 : -1;
    *gradient = (2 * predictedScore) - (2 * expectedScore);
    *hessian = 2;
}
