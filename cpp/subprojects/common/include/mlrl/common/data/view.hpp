/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <utility>

/**
 * A view that provides access to values stored in a pre-allocated array.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
struct View {
        /**
         * A pointer to the array that stores the values, the view provides access to.
         */
        T* array;

        /**
         * @param a A pointer to an array of template type `T` that stores the values, the view should provide access to
         */
        View(T* a) : array(a) {}

        /**
         * @param other A const reference to an object of type `View` that should be copied
         */
        View(const View<T>& other) : array(other.array) {}

        /**
         * @param other A reference to an object of type `View` that should be moved
         */
        View(View<T>&& other) : array(other.array) {}

        virtual ~View() {};

        /**
         * The type of the values, the view provides access to.
         */
        typedef T value_type;

        /**
         * An iterator that provides read-only access to the elements in the view.
         */
        typedef const value_type* const_iterator;

        /**
         * An iterator that provides access to the elements in the view and allows to modify them.
         */
        typedef value_type* iterator;
};
