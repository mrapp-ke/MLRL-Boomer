/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"
#include "boosting/data/statistic_view_label_wise_sparse.hpp"


namespace boosting {

    /**
     * Defines an interface for all (decomposable) loss functions that are applied label-wise and are suited for the use
     * of sparse data structures. To meet this requirement, the gradients and Hessians that are computed by the loss
     * function should be zero, if the prediction for a label is correct.
     */
    class ISparseLabelWiseLoss : public ILabelWiseLoss {

        public:

            virtual ~ISparseLabelWiseLoss() override { };

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousConstView` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `LilMatrix` that stores the currently predicted
             *                          scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const LilMatrix<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousConstView` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `LilMatrix` that stores the currently predicted
             *                          scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const LilMatrix<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `BinaryCsrConstView` that provides row-wise
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `LilMatrix` that stores the currently predicted
             *                          scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const LilMatrix<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `BinaryCsrConstView` that provides row-wise
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `LilMatrix` that stores the currently predicted
             *                          scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const LilMatrix<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

    };

}
