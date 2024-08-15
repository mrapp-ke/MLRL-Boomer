/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/dll_exports.hpp"
#include "mlrl/common/util/memory.hpp"

#include <initializer_list>
#include <stdexcept>
#include <utility>

/**
 * A view that provides access to values stored in a pre-allocated array.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API View {
    public:

        /**
         * A pointer to the array that stores the values, the view provides access to.
         */
        T* array;

        /**
         * @param array         A pointer to an array of template type `T` that stores the values, the view should
         *                      provide access to
         * @param dimensions    The number of elements in each dimension of the view
         */
        View(T* array, std::initializer_list<uint32> dimensions) : array(array) {}

        /**
         * @param array A pointer to an array of template type `T` that stores the values, the view should provide
         *              access to
         */
        explicit View(T* array) : array(array) {}

        /**
         * @param other A const reference to an object of type `View` that should be copied
         */
        View(const View<T>& other) : array(other.array) {}

        /**
         * @param other A reference to an object of type `View` that should be moved
         */
        View(View<T>&& other) : array(other.array) {}

        virtual ~View() {}

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
         * Releases the ownership of the array that stores the values, the view provides access to. As a result, the
         * behavior of this view becomes undefined and it should not be used anymore. The caller is responsible for
         * freeing the memory that is occupied by the array.
         *
         * @return A pointer to the array that stores the values, the view provided access to
         */
        value_type* release() {
            value_type* ptr = array;
            array = nullptr;
            return ptr;
        }

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

        /**
         * Returns a const reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A const reference to the specified element
         */
        const value_type& operator[](uint32 pos) const {
            return array[pos];
        }

        /**
         * Returns a reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A reference to the specified element
         */
        value_type& operator[](uint32 pos) {
            return array[pos];
        }
};

/**
 * Allocates the memory, a view provides access to.
 *
 * @tparam View The type of the view
 */
template<typename View>
class MLRLCOMMON_API Allocator : public View {
    public:

        /**
         * @param numElements   The number of elements in the view
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit Allocator(uint32 numElements, bool init = false)
            : View(util::allocateMemory<typename View::value_type>(numElements, init), {numElements}) {}

        /**
         * @param other A reference to an object of type `Allocator` that should be copied
         */
        Allocator(const Allocator<View>& other) : View(other) {
            throw std::runtime_error("Objects of type Allocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `Allocator` that should be moved
         */
        Allocator(Allocator<View>&& other) : View(std::move(other)) {
            other.release();
        }

        virtual ~Allocator() override {
            util::freeMemory(View::array);
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
class MLRLCOMMON_API ResizableAllocator : public Allocator<View> {
    public:

        /**
         * The maximum number of elements in the view.
         */
        uint32 maxCapacity;

        /**
         * @param numElements   The number of elements in the view
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit ResizableAllocator(uint32 numElements, bool init = false)
            : Allocator<View>(numElements, init), maxCapacity(numElements) {}

        /**
         * @param other A reference to an object of type `ResizableAllocator` that should be copied
         */
        ResizableAllocator(const ResizableAllocator<View>& other)
            : Allocator<View>(other), maxCapacity(other.maxCapacity) {
            throw std::runtime_error("Objects of type ResizableAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `ResizableAllocator` that should be moved
         */
        ResizableAllocator(ResizableAllocator<View>&& other)
            : Allocator<View>(std::move(other)), maxCapacity(other.maxCapacity) {}

        virtual ~ResizableAllocator() override {}

        /**
         * Resizes the view by re-allocating the memory it provides access to.
         *
         * @param numElements   The number of elements to which the view should be resized
         * @param freeMemory    True, if unused memory should be freed, false otherwise
         */
        void resize(uint32 numElements, bool freeMemory) {
            if (numElements < maxCapacity) {
                if (freeMemory) {
                    View::array = util::reallocateMemory(View::array, numElements);
                    maxCapacity = numElements;
                }
            } else if (numElements > maxCapacity) {
                View::array = util::reallocateMemory(View::array, numElements);
                maxCapacity = numElements;
            }

            View::numElements = numElements;
        }
};

/**
 * A base class for all data structures that are backed by a view.
 *
 * @tparam View The type of the view, the data structure is backed by
 */
template<typename View>
class MLRLCOMMON_API ViewDecorator {
    protected:

        /**
         * The view, the data structure is backed by.
         */
        View view;

    public:

        /**
         * @param view The view, the data structure should be backed by
         */
        explicit ViewDecorator(View&& view) : view(std::move(view)) {}

        virtual ~ViewDecorator() {}

        /**
         * The type of the view, the data structure is backed by.
         */
        typedef View view_type;

        /**
         * Returns a const reference to the view, the data structure is backed by.
         *
         * @return A const reference to an object of template type `View`, the data structure is backed by
         */
        const View& getView() const {
            return view;
        }

        /**
         * Returns a reference to the view, the data structure is backed by.
         *
         * @return A reference to an object of template type `View`, the data structure is backed by
         */
        View& getView() {
            return view;
        }
};

/**
 * Provides random access to the values stored in a view.
 *
 * @tparam View The type of the view
 */
template<typename View>
class MLRLCOMMON_API IndexableViewDecorator : public View {
    public:

        /**
         * @param view The view, the view should be backed by
         */
        explicit IndexableViewDecorator(typename View::view_type&& view) : View(std::move(view)) {}

        virtual ~IndexableViewDecorator() override {}

        /**
         * An iterator that provides read-only access to the values stored in the view.
         */
        typedef typename View::view_type::const_iterator const_iterator;

        /**
         * An iterator that provides access to the values stored in the view and allows to modify them.
         */
        typedef typename View::view_type::iterator iterator;

        /**
         * Returns a `const_iterator` to the beginning of the view.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return View::view.cbegin();
        }

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
        const typename View::view_type::value_type& operator[](uint32 pos) const {
            return View::view[pos];
        }

        /**
         * Returns a reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A reference to the specified element
         */
        typename View::view_type::value_type& operator[](uint32 pos) {
            return View::view[pos];
        }
};

/**
 * Allows to set all values stored in a view to zero.
 *
 * @tparam View The type of the view
 */
template<typename View>
class MLRLCOMMON_API ClearableViewDecorator : public View {
    public:

        /**
         * @param view The view, the view should be backed by
         */
        explicit ClearableViewDecorator(typename View::view_type&& view) : View(std::move(view)) {}

        virtual ~ClearableViewDecorator() override {}

        /**
         * Sets all values stored in the view to zero.
         */
        virtual void clear() {
            View::view.clear();
        }
};
