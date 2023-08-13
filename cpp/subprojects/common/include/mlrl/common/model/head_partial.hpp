/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_arrays.hpp"
#include "mlrl/common/model/head.hpp"

/**
 * A head that contains a numerical score for a subset of the available labels.
 */
class MLRLCOMMON_API PartialHead final : public IHead {
    private:

        SparseArraysVector<float64> vector_;

    public:

        /**
         * @param numElements The number of scores that are contained by the head
         */
        PartialHead(uint32 numElements);

        /**
         * An iterator that provides access to the scores that are contained by the head and allows to modify them.
         */
        typedef SparseArraysVector<float64>::value_iterator score_iterator;

        /**
         * An iterator that provides read-only access to the scores that are contained by the head.
         */
        typedef SparseArraysVector<float64>::value_const_iterator score_const_iterator;

        /**
         * An iterator that provides access to the indices, the scores that are contained by the head, correspond to and
         * allows to modify them.
         */
        typedef SparseArraysVector<float64>::index_iterator index_iterator;

        /**
         * An iterator that provides read-only access to the indices, the scores that are contained by the head,
         * correspond to.
         */
        typedef SparseArraysVector<float64>::index_const_iterator index_const_iterator;

        /**
         * Returns the number of scores that are contained by the head.
         *
         * @return The number of scores
         */
        uint32 getNumElements() const;

        /**
         * Returns a `score_iterator` to the beginning of the scores that are contained by the head.
         *
         * @return A `score_iterator` to the beginning
         */
        score_iterator scores_begin();

        /**
         * Returns a `score_iterator` to the end of the scores that are contained by the head.
         *
         * @return A `score_iterator` to the end
         */
        score_iterator scores_end();

        /**
         * Returns a `score_const_iterator` to the beginning of the scores that are contained by the head.
         *
         * @return A `score_const_iterator` to the beginning
         */
        score_const_iterator scores_cbegin() const;

        /**
         * Returns a `score_const_iterator` to the end of the scores that are contained by the head.
         *
         * @return A `score_const_iterator` to the end
         */
        score_const_iterator scores_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the indices, the scores that are contained by the head
         * correspond to.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin();

        /**
         * Returns an `index_iterator` to the end of the indices, the scores that are contained by the head correspond
         * to.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the indices, the scores that are contained by the head
         * correspond to.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices, the scores that are contained by the head,
         * correspond to.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        void visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const override;
};
