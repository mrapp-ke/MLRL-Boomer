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
        typedef SparseArraysVector<float64>::value_iterator value_iterator;

        /**
         * An iterator that provides read-only access to the scores that are contained by the head.
         */
        typedef SparseArraysVector<float64>::value_const_iterator value_const_iterator;

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
         * Returns a `value_iterator` to the beginning of the scores that are contained by the head.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin();

        /**
         * Returns a `value_iterator` to the end of the scores that are contained by the head.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end();

        /**
         * Returns a `value_const_iterator` to the beginning of the scores that are contained by the head.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the scores that are contained by the head.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const;

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
