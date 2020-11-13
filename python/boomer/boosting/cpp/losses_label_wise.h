/**
 * Implements different differentiable loss functions that are applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/input_data.h"
#include "data.h"
#include "data_label_wise.h"


namespace boosting {

    /**
     * Defines an interface for all (decomposable) loss functions that are applied label-wise.
     */
    class ILabelWiseLoss {

        public:

            virtual ~ILabelWiseLoss() { };

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `FullIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `IRandomAccessLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `DenseNumericMatrix` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `FullIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `FullIndexVector::const_iterator` to the end of the label indices
             * @param statisticMatrix   A reference to an object of type `DenseLabelWiseStatisticMatrix` to be updated
             */
            virtual void updateStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                          const DenseNumericMatrix<float64>& scoreMatrix,
                                          FullIndexVector::const_iterator labelIndicesBegin,
                                          FullIndexVector::const_iterator labelIndicesEnd,
                                          DenseLabelWiseStatisticMatrix& statisticMatrix) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `IRandomAccessLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `DenseNumericMatrix` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticMatrix   A reference to an object of type `DenseLabelWiseStatisticMatrix` to be updated
             */
            virtual void updateStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                          const DenseNumericMatrix<float64>& scoreMatrix,
                                          PartialIndexVector::const_iterator labelIndicesBegin,
                                          PartialIndexVector::const_iterator labelIndicesEnd,
                                          DenseLabelWiseStatisticMatrix& statisticMatrix) const = 0;

    };

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

            void updateStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                  const DenseNumericMatrix<float64>& scoreMatrix,
                                  FullIndexVector::const_iterator labelIndicesBegin,
                                  FullIndexVector::const_iterator labelIndicesEnd,
                                  DenseLabelWiseStatisticMatrix& statisticMatrix) const override;

            void updateStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                  const DenseNumericMatrix<float64>& scoreMatrix,
                                  PartialIndexVector::const_iterator labelIndicesBegin,
                                  PartialIndexVector::const_iterator labelIndicesEnd,
                                  DenseLabelWiseStatisticMatrix& statisticMatrix) const override;

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
