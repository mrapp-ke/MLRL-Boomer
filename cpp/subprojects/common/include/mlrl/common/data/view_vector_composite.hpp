/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"

/**
 * A vector that is backed two one-dimensional views.
 *
 * @tparam IndexView    The type of the first view
 * @tparam ValueView    The type of the second view
 */
template<typename FirstView, typename SecondView>
class CompositeVectorDecorator {
    protected:

        /**
         * The first view, the vector is backed by.
         */
        FirstView firstView_;

        /**
         * The second view, the vector is backed by.
         */
        SecondView secondView_;

        /**
         * The type of the first view, the vector is backed by.
         */
        typedef FirstView first_view_type;

        /**
         * The type of the second view, the vector is backed by.
         */
        typedef SecondView second_view_type;

    public:

        /**
         * @param firstView     The first view, the vector should be backed by
         * @param secondView    The second view, the vector should be backed by
         */
        CompositeVectorDecorator(FirstView&& firstView, SecondView&& secondView)
            : firstView_(std::move(firstView)), secondView_(std::move(secondView)) {}

        virtual ~CompositeVectorDecorator() {};
};
