/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/input/label_matrix.h"
#include "../data/matrix_dense_numeric.h"
#include "../data/matrix_dense_label_wise.h"


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
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
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
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                                   const DenseNumericMatrix<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticMatrix& statisticMatrix) const = 0;

    };

    /**
     * A multi-label variant of the squared hinge loss that is applied label-wise.
     */
    class LabelWiseSquaredHingeLossImpl : public AbstractLabelWiseLoss {

        public:

            void updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                          DenseVector<float64>::iterator hessian, bool trueLabel,
                                          float64 predictedScore) const override;

    };

}
