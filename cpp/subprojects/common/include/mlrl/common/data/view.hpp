/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/memory.hpp"

#include <utility>

/**
 * A view that provides access to values stored in a pre-allocated array.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class View {
    public:

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

        /**
         * Returns a `const_iterator` to the beginning of the view.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return array;
        }

        /**
         * Returns an `iterator` to the beginning of the view.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin() {
            return array;
        }
};

/**
 * Allocates the memory, a view provides access to.
 *
 * @tparam View The type of the view
 */
template<typename View>
class Allocator : public View {
    public:

        /**
         * @param numElements   The number of elements in the view
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        Allocator(uint32 numElements, bool init = false)
            : View(allocateMemory<typename View::value_type>(numElements, init), numElements) {}

        /**
         * @param other A reference to an object of type `Allocator` that should be moved
         */
        Allocator(Allocator<View>&& other) : View(std::move(other)) {
            other.array = nullptr;
        }

        virtual ~Allocator() override {
            freeMemory(View::array);
        }
};

/**
 * Allocates the memory, a `View` provides access to
 *
 * @tparam T The type of the values stored in the `View`
 */
template<typename T>
using AllocatedView = Allocator<View<T>>;

/**
 * Allocates the memory, a view provides access to, and allows to resize it afterwards.
 *
 * @tparam View The type of the view
 */
template<typename View>
class ResizableAllocator : public Allocator<View> {
    public:

        /**
         * The maximum number of elements in the view.
         */
        uint32 maxCapacity;

        /**
         * @param numElements   The number of elements in the view
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        ResizableAllocator(uint32 numElements, bool init = false)
            : Allocator<View>(numElements, init), maxCapacity(numElements) {}

        /**
         * @param other A reference to an object of type `ResizableAllocator` that should be moved
         */
        ResizableAllocator(ResizableAllocator<View>&& other)
            : Allocator<View>(std::move(other)), maxCapacity(other.maxCapacity) {}

        /**
         * Resizes the view by re-allocating the memory it provides access to.
         *
         * @param numElements   The number of elements to which the view should be resized
         * @param freeMemory    True, if unused memory should be freed, false otherwise
         */
        void resize(uint32 numElements, bool freeMemory) {
            if (numElements < maxCapacity) {
                if (freeMemory) {
                    View::array = reallocateMemory(View::array, numElements);
                    maxCapacity = numElements;
                }
            } else if (numElements > maxCapacity) {
                View::array = reallocateMemory(View::array, numElements);
                maxCapacity = numElements;
            }

            View::numElements = numElements;
        }

        virtual ~ResizableAllocator() override {};
};

/**
 * A base class for all data structures that are backed by a view.
 *
 * @tparam View The type of the view, the data structure is backed by
 */
template<typename View>
class ViewDecorator {
    public:

        /**
         * The view, the data structure is backed by.
         */
        View view;

        /**
         * @param view The view, the data structure should be backed by
         */
        ViewDecorator(View&& view) : view(std::move(view)) {}

        virtual ~ViewDecorator() {};

        /**
         * The type of the view, the data structure is backed by.
         */
        typedef View view_type;

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
         * An iterator that provides read-only access to the values stored in the view.
         */
        typedef typename View::view_type::const_iterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the view.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return View::view.cbegin();
        }

        /**
         * Returns a const reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A const reference to the specified element
         */
        const typename View::value_type& operator[](uint32 pos) const {
            return View::view.array[pos];
        }
};

/**
 * Provides random write access to the values stored in a view.
 *
 * @tparam View The type of the view
 */
template<typename View>
class WriteAccessibleViewDecorator : public View {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        WriteAccessibleViewDecorator(typename View::view_type&& view) : View(std::move(view)) {}

        virtual ~WriteAccessibleViewDecorator() override {};

        /**
         * An iterator that provides access to the values stored in the view and allows to modify them.
         */
        typedef typename View::view_type::iterator iterator;

        /**
         * Returns an `iterator` to the beginning of the view.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin() {
            return View::view.begin();
        }

        /**
         * Returns a const reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A const reference to the specified element
         */
        const typename View::value_type& operator[](uint32 pos) const {
            return View::view.array[pos];
        }

        /**
         * Returns a reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A reference to the specified element
         */
        typename View::value_type& operator[](uint32 pos) {
            return View::view.array[pos];
        }
};
