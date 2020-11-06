/**
 * Implements different differentiable loss functions that are applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/input_data.h"
#include "../../common/cpp/indices.h"


namespace boosting {

    /**
     * Defines an interface for all (decomposable) loss functions that are applied label-wise.
     */
    // TODO Rename to AbstractLabelWiseLoss
    class ILabelWiseLoss {

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

            virtual ~ILabelWiseLoss() { };

            /**
             * Calculates the gradients and Hessians of the example at a specific index and the labels, whose indices
             * are provided by a `FullIndexVector`, and updates the gradient and Hessian matrix accordingly.
             *
             * @param gradients         A reference to an object of type `DenseMatrix` that stores the gradients
             * @param hessians          A reference to an object of type `DenseMatrix` that stores the Hessians
             * @param predictedScores   A reference to an object of type `DenseMatrix` that stores the scores that are
             *                          currently predicted
             * @param labelMatrix       A reference to an object of type `IRandomAccessLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelIndicesBegin A `FullIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `FullIndexVector::const_iterator` to the end of the label indices
             */
            void updateGradientsAndHessians(DenseMatrix<float64>& gradients, DenseMatrix<float64>& hessians,
                                            const DenseMatrix<float64>& predictedScores,
                                            const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex,
                                            FullIndexVector::const_iterator labelIndicesBegin,
                                            FullIndexVector::const_iterator labelIndicesEnd) const;

            /**
             * Calculates the gradients and Hessians of the example at a specific index and the labels, whose indices
             * are provided by a `PartialIndexVector`, and updates the gradient and Hessian matrix accordingly.
             *
             * @param gradients         A reference to an object of type `DenseMatrix` that stores the gradients
             * @param hessians          A reference to an object of type `DenseMatrix` that stores the Hessians
             * @param predictedScores   A reference to an object of type `DenseMatrix` that stores the scores that are
             * @param labelMatrix       A reference to an object of type `IRandomAccessLabelMatrix` that provides random
             *                          access to the labels of the training examples
             *                          currently predicted
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             */
            void updateGradientsAndHessians(DenseMatrix<float64>& gradients, DenseMatrix<float64>& hessians,
                                            const DenseMatrix<float64>& predictedScores,
                                            const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex,
                                            PartialIndexVector::const_iterator labelIndicesBegin,
                                            PartialIndexVector::const_iterator labelIndicesEnd) const;

    };

    /**
     * A multi-label variant of the logistic loss that is applied label-wise.
     */
    class LabelWiseLogisticLossImpl : public ILabelWiseLoss {

        protected:

            void updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                          DenseVector<float64>::iterator hessian, bool trueLabel,
                                          float64 predictedScore) const override;

    };

    /**
     * A multi-label variant of the squared error loss that is applied label-wise.
     */
    class LabelWiseSquaredErrorLossImpl : public ILabelWiseLoss {

        protected:

            void updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                          DenseVector<float64>::iterator hessian, bool trueLabel,
                                          float64 predictedScore) const override;

    };

}
