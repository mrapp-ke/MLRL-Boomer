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
         * @param n The number of elements in the view
         */
        View(T* a, uint32 n) : array(a) {}

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

/**
 * A base class for all data structures that are backed by a view.
 *
 * @tparam View The type of the view, the data structure is backed by
 */
template<typename View>
class ViewDecorator {
    protected:

        /**
         * The view, the data structure is backed by.
         */
        View view_;

        /**
         * The type of the view, the data structure is backed by.
         */
        typedef View view_type;

    public:

        /**
         * @param view The view, the data structure should be backed by
         */
        ViewDecorator(View&& view) : view_(std::move(view)) {}

        virtual ~ViewDecorator() {};

        /**
         * The type of the values that are stored in the data structure.
         */
        typedef typename View::value_type value_type;
};

/**
 * Provides random read-only access to the values stored in a view.
 *
 * @tparam View The type of the view
 */
template<typename View>
class ReadAccessibleViewDecorator : public View {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        ReadAccessibleViewDecorator(typename View::view_type&& view) : View(std::move(view)) {}

        virtual ~ReadAccessibleViewDecorator() override {};

        /**
         * Returns a const reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A const reference to the specified element
         */
        const typename View::value_type& operator[](uint32 pos) const {
            return View::view_.array[pos];
        }
};
