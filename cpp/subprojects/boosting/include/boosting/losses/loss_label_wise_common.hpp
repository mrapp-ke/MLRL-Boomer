/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"
#include "boosting/math/math.hpp"


namespace boosting {

    /**
     * An abstract base class for all (decomposable) loss functions that are applied label-wise.
     */
    class AbstractLabelWiseLoss : public ILabelWiseLoss {

        protected:

            /**
             * Must be implemented by subclasses in order to update the gradient and Hessian for a single example and
             * label.
             *
             * @param gradient          A `DenseVector::iterator` to the gradient that should be updated
             * @param hessian           A `DenseVector::iterator` to the Hessian that should be updated
             * @param trueLabel         True, if the label is relevant, false otherwise
             * @param predictedScore    The score that is predicted for the label
             */
            virtual void updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                                  DenseVector<float64>::iterator hessian, bool trueLabel,
                                                  float64 predictedScore) const = 0;

            /**
             * Must be implemented by subclasses in order to calculate a numerical score that assesses the quality of
             * the prediction for a single example and label.
             *
             * @param trueLabel         True, if the label is relevant, false otherwise
             * @param predictedScore    The score that is predicted for the label
             * @return                  The numerical score that has been calculated
             */
            virtual float64 evaluate(bool trueLabel, float64 predictedScore) const = 0;

        public:

            virtual ~AbstractLabelWiseLoss() { };

            void updateLabelWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                           const DenseNumericMatrix<float64>& scoreMatrix,
                                           FullIndexVector::const_iterator labelIndicesBegin,
                                           FullIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticMatrix& statisticMatrix) const override final {
                DenseLabelWiseStatisticMatrix::gradient_iterator gradientIterator =
                    statisticMatrix.gradients_row_begin(exampleIndex);
                DenseLabelWiseStatisticMatrix::hessian_iterator hessianIterator =
                    statisticMatrix.hessians_row_begin(exampleIndex);
                DenseNumericMatrix<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
                uint32 numLabels = labelMatrix.getNumCols();

                for (uint32 i = 0; i < numLabels; i++) {
                    bool trueLabel = labelMatrix.getValue(exampleIndex, i);
                    float64 predictedScore = scoreIterator[i];
                    this->updateGradientAndHessian(&gradientIterator[i], &hessianIterator[i], trueLabel,
                                                   predictedScore);
                }
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                           const DenseNumericMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticMatrix& statisticMatrix) const override final {
                DenseLabelWiseStatisticMatrix::gradient_iterator gradientIterator =
                    statisticMatrix.gradients_row_begin(exampleIndex);
                DenseLabelWiseStatisticMatrix::hessian_iterator hessianIterator =
                    statisticMatrix.hessians_row_begin(exampleIndex);
                DenseNumericMatrix<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);

                for (auto indexIterator = labelIndicesBegin; indexIterator != labelIndicesEnd; indexIterator++) {
                    uint32 labelIndex = *indexIterator;
                    bool trueLabel = labelMatrix.getValue(exampleIndex, labelIndex);
                    float64 predictedScore = scoreIterator[labelIndex];
                    this->updateGradientAndHessian(&gradientIterator[labelIndex], &hessianIterator[labelIndex],
                                                   trueLabel, predictedScore);
                }
            }

            float64 evaluate(uint32 exampleIndex, const LabelVector& labelVector,
                             const CContiguousView<float64>& scoreMatrix) const override final {
                uint32 numLabels = scoreMatrix.getNumCols();
                CContiguousView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
                LabelVector::index_const_iterator indexIterator = labelVector.indices_cbegin();
                LabelVector::index_const_iterator indicesEnd = labelVector.indices_cend();
                float64 mean = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoreIterator[i];
                    bool trueLabel;

                    if (indexIterator != indicesEnd && *indexIterator == i) {
                        indexIterator++;
                        trueLabel = true;
                    } else {
                        trueLabel = false;
                    }

                    float64 score = this->evaluate(trueLabel, predictedScore);
                    mean = iterativeMean<float64>(i + 1, score, mean);
                }

                return mean;
            }

    };

}
