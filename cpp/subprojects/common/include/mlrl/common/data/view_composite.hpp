/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

/**
 * A view that is backed two views.
 *
 * @tparam IndexView    The type of the first view
 * @tparam ValueView    The type of the second view
 */
template<typename FirstView, typename SecondView>
class CompositeViewDecorator {
    protected:

        /**
         * The first view, the view is backed by.
         */
        FirstView firstView_;

        /**
         * The second view, the view is backed by.
         */
        SecondView secondView_;

        /**
         * The type of the first view, the view is backed by.
         */
        typedef FirstView first_view_type;

        /**
         * The type of the second view, the view is backed by.
         */
        typedef SecondView second_view_type;

    public:

        /**
         * @param firstView     The first view, the vector should be backed by
         * @param secondView    The second view, the vector should be backed by
         */
        CompositeViewDecorator(FirstView&& firstView, SecondView&& secondView)
            : firstView_(std::move(firstView)), secondView_(std::move(secondView)) {}

        virtual ~CompositeViewDecorator() {}
};
