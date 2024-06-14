/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"
#include "mlrl/common/model/head.hpp"

/**
 * A head that contains a numerical score for each available output.
 */
class MLRLCOMMON_API CompleteHead final : public VectorDecorator<AllocatedVector<float64>>,
                                          public IHead {
    public:

        /**
         * @param numElements The number of scores that are contained by the head.
         */
        CompleteHead(uint32 numElements);

        /**
         * An iterator that provides access to the scores the are contained by the head and allows to modify them.
         */
        typedef View<float64>::iterator value_iterator;

        /**
         * An iterator that provides read-only access to the scores that are contained by the head.
         */
        typedef View<float64>::const_iterator value_const_iterator;

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

        void visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const override;
};
