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

/**
 * Allows to set all values stored in a vector that is backed by two one-dimensional views to zero.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class ClearableCompositeVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        ClearableCompositeVectorDecorator(typename Vector::first_view_type&& firstView,
                                          typename Vector::second_view_type&& secondView)
            : Vector(std::move(firstView), std::move(secondView)) {}

        /**
         * Sets all values stored in the vector to zero.
         */
        virtual void clear() {
            setViewToZeros(this->firstView_.array, this->firstView_.numElements);
            setViewToZeros(this->secondView_.array, this->secondView_.numElements);
        }
};
