/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_binned.hpp"
#include "mlrl/common/iterator/iterator_binned.hpp"
#include "mlrl/common/rule_evaluation/score_vector.hpp"

/**
 * An one dimensional vector that stores the scores that may be predicted by a rule, corresponding to bins for which the
 * same prediction is made, as well as a numerical score that assesses the overall quality of the rule, in a
 * C-contiguous array.
 *
 * @tparam IndexVector The type of the vector that provides access to the indices of the outputs for which the rule may
 *                     predict
 */
template<typename IndexVector>
class DenseBinnedScoreVector final
    : public BinnedVectorDecorator<ViewDecorator<CompositeVector<AllocatedVector<uint32>, ResizableVector<float64>>>>,
      virtual public IScoreVector {
    private:

        const IndexVector& outputIndices_;

        const bool sorted_;

        uint32 maxCapacity_;

    public:

        /**
         * @param outputIndices A reference to an object of template type `IndexVector` that provides access to the
         *                      indices of the outputs for which the rule may predict
         * @param numBins       The number of bins
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         */
        DenseBinnedScoreVector(const IndexVector& outputIndices, uint32 numBins, bool sorted);

        /**
         * An iterator that provides read-only access to the indices of the output for which the rule predicts.
         */
        typedef typename IndexVector::const_iterator index_const_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores that correspond to individual outputs.
         */
        typedef BinnedIterator<const float64> value_const_iterator;

        /**
         * An iterator that provides access to the indices that correspond to individual bins and allows to modify them.
         */
        typedef typename View<uint32>::iterator bin_index_iterator;

        /**
         * An iterator that provides read-only access to the indices that correspond to individual bins.
         */
        typedef typename View<uint32>::const_iterator bin_index_const_iterator;

        /**
         * An iterator that provides access to the predicted scores that correspond to individual bins and allows to
         * modify them.
         */
        typedef typename View<float64>::iterator bin_value_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores that correspond to individual bins.
         */
        typedef typename View<float64>::const_iterator bin_value_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices that correspond to individual outputs.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices that correspond to individual outputs.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns a `value_const_iterator` to the beginning of the predicted scores that correspond to individual
         * outputs.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the predicted scores that correspond to individual outputs.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const;

        /**
         * Returns an `bin_index_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `bin_index_iterator` to the beginning
         */
        bin_index_iterator bin_indices_begin();

        /**
         * Returns an `bin_index_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `bin_index_iterator` to the end
         */
        bin_index_iterator bin_indices_end();

        /**
         * Returns an `bin_index_const_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `bin_index_const_iterator` to the beginning
         */
        bin_index_const_iterator bin_indices_cbegin() const;

        /**
         * Returns an `bin_index_const_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `bin_index_const_iterator` to the end
         */
        bin_index_const_iterator bin_indices_cend() const;

        /**
         * Returns a `bin_value_iterator` to the beginning of the predicted scores that correspond to individual bins.
         *
         * @return A `bin_value_iterator` to the beginning
         */
        bin_value_iterator bin_values_begin();

        /**
         * Returns a `bin_value_iterator` to the end of the predicted scores that correspond to individual bins.
         *
         * @return A `bin_value_iterator` to the end
         */
        bin_value_iterator bin_values_end();

        /**
         * Returns a `bin_value_const_iterator` to the beginning of the predicted scores that correspond to individual
         * bins.
         *
         * @return A `bin_value_const_iterator` to the beginning
         */
        bin_value_const_iterator bin_values_cbegin() const;

        /**
         * Returns a `bin_value_const_iterator` to the end of the predicted scores that correspond to individual bins.
         *
         * @return A `bin_value_const_iterator` to the end
         */
        bin_value_const_iterator bin_values_cend() const;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements
         */
        uint32 getNumElements() const;

        /**
         * Sets the number of bins in the vector.
         *
         * @param numBins       The number of bins to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumBins(uint32 numBins, bool freeMemory);

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

        void visit(DenseVisitor<CompleteIndexVector> completeDenseVisitor,
                   DenseVisitor<PartialIndexVector> partialDenseVisitor,
                   DenseBinnedVisitor<CompleteIndexVector> completeDenseBinnedVisitor,
                   DenseBinnedVisitor<PartialIndexVector> partialDenseBinnedVisitor) const override;

        void updatePrediction(IPrediction& prediction) const override;
};
