/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"
#include "mlrl/common/rule_evaluation/score_vector.hpp"

/**
 * An one-dimensional vector that stores the scores that may be predicted by a rule, as well as an overall quality
 * score that assesses the overall quality of the rule, in a C-contiguous array.
 *
 * @tparam IndexVector The type of the vector that provides access to the indices of the outputs for which the rule may
 *                     predict
 */
template<typename IndexVector>
class DenseScoreVector final : public ViewDecorator<AllocatedView<float64>>,
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
        DenseScoreVector(const IndexVector& outputIndices, bool sorted);

        /**
         * An iterator that provides read-only access to the indices.
         */
        typedef typename IndexVector::const_iterator index_const_iterator;

        /**
         * An iterator that provides access to the predicted scores and allows to modify them.
         */
        typedef View<float64>::iterator value_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores.
         */
        typedef View<float64>::const_iterator value_const_iterator;

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
         * Returns a `value_iterator` to the beginning of the predicted scores.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin();

        /**
         * Returns a `value_iterator` to the end of the predicted scores.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end();

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
         * @return The number of outputs
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

        void updatePrediction(IPrediction& prediction) const override;

        void processScores(ScoreProcessor& scoreProcessor) const override;
};
