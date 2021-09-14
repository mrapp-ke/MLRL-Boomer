/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"
#include "boosting/data/statistic_view_label_wise_sparse.hpp"
#include "common/measures/measure_evaluation_sparse.hpp"


namespace boosting {

    /**
     * Defines an interface for all (decomposable) loss functions that are applied label-wise and are suited for the use
     * of sparse data structures. To meet this requirement, the gradients and Hessians that are computed by the loss
     * function should be zero, if the prediction for a label is correct.
     */
    class ISparseLabelWiseLoss : virtual public ILabelWiseLoss, virtual public ISparseEvaluationMeasure {

        public:

            virtual ~ISparseLabelWiseLoss() { };

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `LilMatrix` that stores the currently predicted
             *                          scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                   const LilMatrix<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `LilMatrix` that stores the currently predicted
             *                          scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                   const LilMatrix<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CsrLabelMatrix` that provides row-wise access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `LilMatrix` that stores the currently predicted
             *                          scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                   const LilMatrix<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CsrLabelMatrix` that provides row-wise access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `LilMatrix` that stores the currently predicted
             *                          scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                   const LilMatrix<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

    };

    /**
     * An abstract base class for all (decomposable) loss functions that are applied label-wise and are suited for the
     * use of sparse data structures.
     */
    class AbstractSparseLabelWiseLoss : public AbstractLabelWiseLoss, virtual public ISparseLabelWiseLoss {

        protected:

            /**
             * @param updateFunction    The function to be used for updating gradients and Hessians
             * @param evaluateFunction  The function to be used for evaluating predictions
             */
            AbstractSparseLabelWiseLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction);

        public:

            virtual ~AbstractSparseLabelWiseLoss() { };

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override;

            float64 evaluate(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                             const LilMatrix<float64>& scoreMatrix) const override final;

            float64 evaluate(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                             const LilMatrix<float64>& scoreMatrix) const override final;

    };

}
