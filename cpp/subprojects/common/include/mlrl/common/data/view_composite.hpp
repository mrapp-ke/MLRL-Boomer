/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <utility>

/**
 * A view that is backed two one-dimensional views.
 *
 * @tparam FirstView    The type of the first view
 * @tparam SecondView   The type of the second view
 */
template<typename FirstView, typename SecondView>
class MLRLCOMMON_API CompositeView {
    public:

        /**
         * The first view, the view is backed by.
         */
        FirstView firstView;

        /**
         * The second view, the view is backed by.
         */
        SecondView secondView;

        /**
         * @param firstView     The first view, the view should be backed by
         * @param secondView    The second view, the view should be backed by
         */
        CompositeView(FirstView&& firstView, SecondView&& secondView)
            : firstView(std::move(firstView)), secondView(std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `CompositeView` that should be copied
         */
        CompositeView(const CompositeView<FirstView, SecondView>& other)
            : firstView(other.firstView), secondView(other.secondView) {}

        /**
         * @param other A reference to an object of type `CompositeView` that should be moved
         */
        CompositeView(CompositeView<FirstView, SecondView>&& other)
            : firstView(std::move(other.firstView)), secondView(std::move(other.secondView)) {}

        virtual ~CompositeView() {}

        /**
         * The type of the first view, the view is backed by.
         */
        using first_view_type = FirstView;

        /**
         * The type of the second view, the view is backed by.
         */
        using second_view_type = SecondView;
};
