#include "losses_label_wise.h"
#include <cmath>

using namespace boosting;


void ILabelWiseLoss::updateGradientsAndHessians(DenseMatrix<float64>& gradients, DenseMatrix<float64>& hessians,
                                                const DenseMatrix<float64>& predictedScores,
                                                const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex,
                                                const FullIndexVector::const_iterator labelIndicesBegin,
                                                const FullIndexVector::const_iterator labelIndicesEnd) const {
    uint32 numLabels = labelMatrix.getNumLabels();
    DenseVector<float64>::iterator gradientIterator = gradients.row_begin(exampleIndex);
    DenseVector<float64>::iterator hessianIterator = hessians.row_begin(exampleIndex);
    DenseVector<float64>::const_iterator scoreIterator = predictedScores.row_cbegin(exampleIndex);

    for (uint32 i = 0; i < numLabels; i++) {
        bool trueLabel = labelMatrix.getValue(exampleIndex, i);
        float64 predictedScore = scoreIterator[i];
        this->updateGradientAndHessian(&gradientIterator[i], &hessianIterator[i], trueLabel, predictedScore);
    }
}

void ILabelWiseLoss::updateGradientsAndHessians(DenseMatrix<float64>& gradients, DenseMatrix<float64>& hessians,
                                                const DenseMatrix<float64>& predictedScores,
                                                const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex,
                                                const PartialIndexVector::const_iterator labelIndicesBegin,
                                                const PartialIndexVector::const_iterator labelIndicesEnd) const {
    DenseVector<float64>::iterator gradientIterator = gradients.row_begin(exampleIndex);
    DenseVector<float64>::iterator hessianIterator = hessians.row_begin(exampleIndex);
    DenseVector<float64>::const_iterator scoreIterator = predictedScores.row_cbegin(exampleIndex);

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
