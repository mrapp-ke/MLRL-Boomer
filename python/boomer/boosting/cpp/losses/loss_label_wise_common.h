/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "loss_label_wise.h"


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

        public:

            virtual ~AbstractLabelWiseLoss() { };

            void updateLabelWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                           const DenseNumericMatrix<float64>& scoreMatrix,
                                           FullIndexVector::const_iterator labelIndicesBegin,
                                           FullIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticMatrix& statisticMatrix) const override;

            void updateLabelWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                           const DenseNumericMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticMatrix& statisticMatrix) const override;

    };

}
