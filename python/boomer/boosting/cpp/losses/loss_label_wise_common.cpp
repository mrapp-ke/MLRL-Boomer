#include "loss_label_wise_common.h"

using namespace boosting;


void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                                      const DenseNumericMatrix<float64>& scoreMatrix,
                                                      const FullIndexVector::const_iterator labelIndicesBegin,
                                                      const FullIndexVector::const_iterator labelIndicesEnd,
                                                      DenseLabelWiseStatisticMatrix& statisticMatrix) const {
    DenseLabelWiseStatisticMatrix::gradient_iterator gradientIterator =
        statisticMatrix.gradients_row_begin(exampleIndex);
    DenseLabelWiseStatisticMatrix::hessian_iterator hessianIterator = statisticMatrix.hessians_row_begin(exampleIndex);
    DenseNumericMatrix<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.getNumLabels();

    for (uint32 i = 0; i < numLabels; i++) {
        bool trueLabel = labelMatrix.getValue(exampleIndex, i);
        float64 predictedScore = scoreIterator[i];
        this->updateGradientAndHessian(&gradientIterator[i], &hessianIterator[i], trueLabel, predictedScore);
    }
}

void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                                      const DenseNumericMatrix<float64>& scoreMatrix,
                                                      const PartialIndexVector::const_iterator labelIndicesBegin,
                                                      const PartialIndexVector::const_iterator labelIndicesEnd,
                                                      DenseLabelWiseStatisticMatrix& statisticMatrix) const {
    DenseLabelWiseStatisticMatrix::gradient_iterator gradientIterator =
        statisticMatrix.gradients_row_begin(exampleIndex);
    DenseLabelWiseStatisticMatrix::hessian_iterator hessianIterator = statisticMatrix.hessians_row_begin(exampleIndex);
    DenseNumericMatrix<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);

    for (auto indexIterator = labelIndicesBegin; indexIterator != labelIndicesEnd; indexIterator++) {
        uint32 labelIndex = *indexIterator;
        bool trueLabel = labelMatrix.getValue(exampleIndex, labelIndex);
        float64 predictedScore = scoreIterator[labelIndex];
        this->updateGradientAndHessian(&gradientIterator[labelIndex], &hessianIterator[labelIndex], trueLabel,
                                       predictedScore);
    }
}

void LabelWiseSquaredErrorLossImpl::updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                                             DenseVector<float64>::iterator hessian, bool trueLabel,
                                                             float64 predictedScore) const {
    if (trueLabel) {
        if (predictedScore < 1) {
            *gradient = 2 * (predictedScore - 1);
            *hessian = 2;
        } else {
            *gradient = 0;
            *hessian = 0;
        }
    } else {
        if (predictedScore > 0) {
            *gradient = 2 * predictedScore;
            *hessian = 2;
        } else {
            *gradient = 0;
            *hessian = 0;
        }
    }
}
