/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_bit.hpp"
#include "mlrl/common/rule_evaluation/score_vector.hpp"

/**
 * An one-dimensional vector that stores binary scores that may be predicted by a rule, as well as an overall quality
 * score that assesses the overall quality of the rule, in a space efficient way.
 *
 * @tparam IndexVector The type of the vector that provides access to the indices of the outputs for which the rule may
 *                     predict
 */
template<typename IndexVector>
class BitScoreVector final : public IndexableBitVectorDecorator<ViewDecorator<AllocatedBitVector>>,
                             virtual public IScoreVector {
    private:

        const IndexVector& outputIndices_;

        const bool sorted_;

    public:

        /**
         * @param outputIndices A reference to an object of template type `IndexVector` that provides access to the
         *                      indices of the outputs for which the rule may predict
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         */
        BitScoreVector(const IndexVector& outputIndices, bool sorted);

        /**
         * The type of the predicted scores that are stored by the vector.
         */
        typedef uint8 value_type;

        /**
         * An iterator that provides read-only access to the indices.
         */
        typedef typename IndexVector::const_iterator index_const_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores.
         */
        typedef typename BitView::bit_const_iterator value_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns a `value_const_iterator` to the beginning of the predicted scores.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the predicted scores.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const;

        /**
         * Returns the number of outputs for which the rule may predict.
         *
         * @return The number of outputs for which the rule may predict
         */
        uint32 getNumElements() const;

        /**
         * Returns whether the rule may only predict for a subset of the available outputs, or not.
         *
         * @return True, if the rule may only predict for a subset of the available outputs, false otherwise
         */
        bool isPartial() const;

        /**
         * Returns whether the indices of the outputs for which the rule may predict are sorted in increasing order, or
         * not.
         *
         * @return True, if the indices of the outputs for which the rule may predict are sorted in increasing order,
         *         false otherwise
         */
        bool isSorted() const;

        void visit(BitVisitor<CompleteIndexVector> completeBitVisitor, BitVisitor<PartialIndexVector> partialBitVisitor,
                   DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
                   DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
                   DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
                   DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
                   DenseBinnedVisitor<float32, CompleteIndexVector> completeDenseBinned32BitVisitor,
                   DenseBinnedVisitor<float32, PartialIndexVector> partialDenseBinned32BitVisitor,
                   DenseBinnedVisitor<float64, CompleteIndexVector> completeDenseBinned64BitVisitor,
                   DenseBinnedVisitor<float64, PartialIndexVector> partialDenseBinned64BitVisitor) const override;
};
