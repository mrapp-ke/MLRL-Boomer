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
     * An abstract base class for all (decomposable) loss functions that are applied label-wise.
     */
    class AbstractLabelWiseLoss {

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

            /**
             * Calculates the gradients and Hessians of the example at a specific index and the labels, whose indices
             * are provided by a `FullIndexVector`, and updates the gradient and Hessian matrix accordingly.
             *
             * @param gradientsBegin    TODO
             * @param gradientsEnd      TODO
             * @param hessiansBegin     TODO
             * @param hessiansEnd       TODO
             * @param scoresBegin       TODO
             * @param scoresEnd         TODO
             * @param labelMatrix       A reference to an object of type `IRandomAccessLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelIndicesBegin A `FullIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `FullIndexVector::const_iterator` to the end of the label indices
             */
            void updateGradientsAndHessians(DenseVector<float64>::iterator gradientsBegin,
                                            DenseVector<float64>::iterator gradientsEnd,
                                            DenseVector<float64>::iterator hessiansBegin,
                                            DenseVector<float64>::iterator hessiansEnd,
                                            DenseVector<float64>::const_iterator scoresBegin,
                                            DenseVector<float64>::const_iterator scoresEnd,
                                            const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex,
                                            FullIndexVector::const_iterator labelIndicesBegin,
                                            FullIndexVector::const_iterator labelIndicesEnd) const;

            /**
             * Calculates the gradients and Hessians of the example at a specific index and the labels, whose indices
             * are provided by a `PartialIndexVector`, and updates the gradient and Hessian matrix accordingly.
             *
             * @param gradientsBegin    TODO
             * @param gradientsEnd      TODO
             * @param hessiansBegin     TODO
             * @param hessiansEnd       TODO
             * @param scoresBegin       TODO
             * @param scoresEnd         TODO
             * @param labelMatrix       A reference to an object of type `IRandomAccessLabelMatrix` that provides random
             *                          access to the labels of the training examples
             *                          currently predicted
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             */
            void updateGradientsAndHessians(DenseVector<float64>::iterator gradientsBegin,
                                            DenseVector<float64>::iterator gradientsEnd,
                                            DenseVector<float64>::iterator hessiansBegin,
                                            DenseVector<float64>::iterator hessiansEnd,
                                            DenseVector<float64>::const_iterator scoresBegin,
                                            DenseVector<float64>::const_iterator scoresEnd,
                                            const IRandomAccessLabelMatrix& labelMatrix, uint32 exampleIndex,
                                            PartialIndexVector::const_iterator labelIndicesBegin,
                                            PartialIndexVector::const_iterator labelIndicesEnd) const;

    };

    /**
     * A multi-label variant of the logistic loss that is applied label-wise.
     */
    class LabelWiseLogisticLossImpl : public AbstractLabelWiseLoss {

        protected:

            void updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                          DenseVector<float64>::iterator hessian, bool trueLabel,
                                          float64 predictedScore) const override;

    };

    /**
     * A multi-label variant of the squared error loss that is applied label-wise.
     */
    class LabelWiseSquaredErrorLossImpl : public AbstractLabelWiseLoss {

        protected:

            void updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                          DenseVector<float64>::iterator hessian, bool trueLabel,
                                          float64 predictedScore) const override;

    };

}
