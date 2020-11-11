#include "losses_label_wise.h"
#include <cmath>

using namespace boosting;


void AbstractLabelWiseLoss::updateGradientsAndHessians(DenseVector<float64>::iterator gradientsBegin,
                                                       DenseVector<float64>::iterator gradientsEnd,
                                                       DenseVector<float64>::iterator hessiansBegin,
                                                       DenseVector<float64>::iterator hessiansEnd,
                                                       DenseVector<float64>::const_iterator scoresBegin,
                                                       DenseVector<float64>::const_iterator scoresEnd,
                                                       const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex,
                                                       const FullIndexVector::const_iterator labelIndicesBegin,
                                                       const FullIndexVector::const_iterator labelIndicesEnd) const {
    uint32 numLabels = labelMatrix.getNumLabels();

    for (uint32 i = 0; i < numLabels; i++) {
        bool trueLabel = labelMatrix.getValue(exampleIndex, i);
        float64 predictedScore = scoresBegin[i];
        this->updateGradientAndHessian(&gradientsBegin[i], &hessiansBegin[i], trueLabel, predictedScore);
    }
}

void AbstractLabelWiseLoss::updateGradientsAndHessians(DenseVector<float64>::iterator gradientsBegin,
                                                       DenseVector<float64>::iterator gradientsEnd,
                                                       DenseVector<float64>::iterator hessiansBegin,
                                                       DenseVector<float64>::iterator hessiansEnd,
                                                       DenseVector<float64>::const_iterator scoresBegin,
                                                       DenseVector<float64>::const_iterator scoresEnd,
                                                       const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex,
                                                       const PartialIndexVector::const_iterator labelIndicesBegin,
                                                       const PartialIndexVector::const_iterator labelIndicesEnd) const {
    for (auto indexIterator = labelIndicesBegin; indexIterator != labelIndicesEnd; indexIterator++) {
        uint32 labelIndex = *indexIterator;
        bool trueLabel = labelMatrix.getValue(exampleIndex, labelIndex);
        float64 predictedScore = scoresBegin[labelIndex];
        this->updateGradientAndHessian(&gradientsBegin[labelIndex], &hessiansBegin[labelIndex], trueLabel,
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
