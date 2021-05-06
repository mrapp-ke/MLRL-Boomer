/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector_full.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/measures/measure_evaluation.hpp"
#include "common/measures/measure_similarity.hpp"
#include "boosting/data/statistic_view_dense_label_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all (decomposable) loss functions that are applied label-wise.
     */
    class ILabelWiseLoss : public IEvaluationMeasure, public ISimilarityMeasure {

        public:

            virtual ~ILabelWiseLoss() { };

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `FullIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `FullIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `FullIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   FullIndexVector::const_iterator labelIndicesBegin,
                                                   FullIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `FullIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CsrLabelMatrix` that provides row-wise access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   FullIndexVector::const_iterator labelIndicesBegin,
                                                   FullIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CsrLabelMatrix` that provides row-wise access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64> scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

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

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           FullIndexVector::const_iterator labelIndicesBegin,
                                           FullIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           FullIndexVector::const_iterator labelIndicesBegin,
                                           FullIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                            const CContiguousConstView<float64> scoreMatrix,
                                            PartialIndexVector::const_iterator labelIndicesBegin,
                                            PartialIndexVector::const_iterator labelIndicesEnd,
                                            DenseLabelWiseStatisticView& statisticView) const override final;

            float64 evaluate(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override final;

            float64 evaluate(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override final;

            float64 measureSimilarity(const LabelVector& labelVector,
                                      CContiguousView<float64>::const_iterator scoresBegin,
                                      CContiguousView<float64>::const_iterator scoresEnd) const override final;

    };

}
