#include "boosting/losses/loss_label_wise.hpp"
#include "common/math/math.hpp"


namespace boosting {

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                          const CContiguousLabelMatrix& labelMatrix,
                                                          const CContiguousConstView<float64>& scoreMatrix,
                                                          FullIndexVector::const_iterator labelIndicesBegin,
                                                          FullIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticView& statisticView) const {
        DenseLabelWiseStatisticView::gradient_iterator gradientIterator =
            statisticView.gradients_row_begin(exampleIndex);
        DenseLabelWiseStatisticView::hessian_iterator hessianIterator =
            statisticView.hessians_row_begin(exampleIndex);
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelMatrix.getNumCols();

        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = labelIterator[i];
            float64 predictedScore = scoreIterator[i];
            this->updateGradientAndHessian(trueLabel, predictedScore, &gradientIterator[i], &hessianIterator[i]);
        }
    }

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                          const CContiguousLabelMatrix& labelMatrix,
                                                          const CContiguousConstView<float64>& scoreMatrix,
                                                          PartialIndexVector::const_iterator labelIndicesBegin,
                                                          PartialIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticView& statisticView) const {
        DenseLabelWiseStatisticView::gradient_iterator gradientIterator =
            statisticView.gradients_row_begin(exampleIndex);
        DenseLabelWiseStatisticView::hessian_iterator hessianIterator =
            statisticView.hessians_row_begin(exampleIndex);
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelIndicesEnd - labelIndicesBegin;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 labelIndex = labelIndicesBegin[i];
            bool trueLabel = labelIterator[labelIndex];
            float64 predictedScore = scoreIterator[labelIndex];
            this->updateGradientAndHessian(trueLabel, predictedScore, &gradientIterator[labelIndex],
                                           &hessianIterator[labelIndex]);
        }
    }

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                          const CsrLabelMatrix& labelMatrix,
                                                          const CContiguousConstView<float64>& scoreMatrix,
                                                          FullIndexVector::const_iterator labelIndicesBegin,
                                                          FullIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticView& statisticView) const {
        DenseLabelWiseStatisticView::gradient_iterator gradientIterator =
            statisticView.gradients_row_begin(exampleIndex);
        DenseLabelWiseStatisticView::hessian_iterator hessianIterator =
            statisticView.hessians_row_begin(exampleIndex);
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CsrLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelMatrix.getNumCols();

        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = *labelIterator;
            float64 predictedScore = scoreIterator[i];
            this->updateGradientAndHessian(trueLabel, predictedScore, &gradientIterator[i], &hessianIterator[i]);
            labelIterator++;
        }
    }

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                          const CContiguousConstView<float64> scoreMatrix,
                                                          PartialIndexVector::const_iterator labelIndicesBegin,
                                                          PartialIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticView& statisticView) const {
        DenseLabelWiseStatisticView::gradient_iterator gradientIterator =
            statisticView.gradients_row_begin(exampleIndex);
        DenseLabelWiseStatisticView::hessian_iterator hessianIterator =
            statisticView.hessians_row_begin(exampleIndex);
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CsrLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelIndicesEnd - labelIndicesBegin;
        uint32 previousLabelIndex = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 labelIndex = labelIndicesBegin[i];
            std::advance(labelIterator, labelIndex - previousLabelIndex);
            bool trueLabel = *labelIterator;
            float64 predictedScore = scoreIterator[labelIndex];
            this->updateGradientAndHessian(trueLabel, predictedScore, &gradientIterator[labelIndex],
                                           &hessianIterator[labelIndex]);
            previousLabelIndex = labelIndex;
        }
    }

    float64 AbstractLabelWiseLoss::evaluate(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                            const CContiguousConstView<float64>& scoreMatrix) const {
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelMatrix.getNumCols();
        float64 mean = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = labelIterator[i];
            float64 score = this->evaluate(trueLabel, predictedScore);
            mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
        }

        return mean;
    }

    float64 AbstractLabelWiseLoss::evaluate(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                            const CContiguousConstView<float64>& scoreMatrix) const {
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CsrLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelMatrix.getNumCols();
        float64 mean = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel= *labelIterator;
            float64 score = this->evaluate(trueLabel, predictedScore);
            mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
            labelIterator++;
        }

        return mean;
    }

    float64 AbstractLabelWiseLoss::measureSimilarity(const LabelVector& labelVector,
                                                     CContiguousConstView<float64>::const_iterator scoresBegin,
                                                     CContiguousConstView<float64>::const_iterator scoresEnd) const {
        uint32 numLabels = scoresEnd - scoresBegin;
        LabelVector::value_const_iterator labelIterator = labelVector.values_cbegin();
        float64 mean = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoresBegin[i];
            bool trueLabel = *labelIterator;
            float64 score = this->evaluate(trueLabel, predictedScore);
            mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
            labelIterator++;
        }

        return mean;
    }

}
