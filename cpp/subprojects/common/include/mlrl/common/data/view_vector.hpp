/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"
#include "mlrl/common/util/view_functions.hpp"

/**
 * A one-dimensional view that provides access to values stored in a pre-allocated array of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class Vector : public View<T> {
    public:

        /**
         * The number of elements in the view.
         */
        uint32 numElements;

        /**
         * @param a A pointer to an array of template type `T` that stores the values, the view should provide access to
         * @param n The number of elements in the view
         */
        Vector(T* a, uint32 n) : View<T>(a, n), numElements(n) {}

        /**
         * @param other A const reference to an object of type `Vector` that should be copied
         */
        Vector(const Vector<T>& other) : Vector(other.array, other.numElements) {}

        /**
         * @param other A reference to an object of type `Vector` that should be moved
         */
        Vector(Vector<T>&& other) : Vector(other.array, other.numElements) {}

        virtual ~Vector() override {}

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        typename View<T>::const_iterator cend() const {
            return &View<T>::array[numElements];
        }

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        typename View<T>::iterator end() {
            return &View<T>::array[numElements];
        }
};

/**
 * Allocates the memory, a `Vector` provides access to
 *
 * @tparam T The type of the values stored in the `Vector`
 */
template<typename T>
using AllocatedVector = Allocator<Vector<T>>;

/**
 * Allocates the memory, a `Vector` provides access to, and allows to resize it afterwards.
 *
 * @tparam T The type of the values stored in the `Vector`
 */
template<typename T>
using ResizableVector = ResizableAllocator<Vector<T>>;

/**
 * A vector that is backed by a one-dimensional view of a specific size.
 *
 * @tparam View The type of view, the vector is backed by
 */
template<typename View>
class VectorDecorator : public ViewDecorator<View> {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        VectorDecorator(View&& view) : ViewDecorator<View>(std::move(view)) {}

        virtual ~VectorDecorator() override {}

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const {
            return ViewDecorator<View>::view.numElements;
        }
};

/**
 * Provides read-only access via iterators to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class ReadIterableVectorDecorator : public ReadAccessibleViewDecorator<Vector> {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        ReadIterableVectorDecorator(typename Vector::view_type&& view)
            : ReadAccessibleViewDecorator<Vector>(std::move(view)) {}

        virtual ~ReadIterableVectorDecorator() override {}

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        typename ReadAccessibleViewDecorator<Vector>::const_iterator cend() const {
            return Vector::view.cend();
        }
};

/**
 * Provides write access via iterators to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class WriteIterableVectorDecorator : public WriteAccessibleViewDecorator<Vector> {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        WriteIterableVectorDecorator(typename Vector::view_type&& view)
            : WriteAccessibleViewDecorator<Vector>(std::move(view)) {}

        virtual ~WriteIterableVectorDecorator() override {}

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        typename WriteAccessibleViewDecorator<Vector>::iterator end() {
            return Vector::view.end();
        }
};

/**
 * Provides random read-only access, as well as read-only access via iterators, to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
using ReadableVectorDecorator = ReadIterableVectorDecorator<VectorDecorator<Vector>>;

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values stored in a
 * vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
using WritableVectorDecorator = WriteIterableVectorDecorator<ReadableVectorDecorator<Vector>>;

/**
 * Allows to resize a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class ResizableVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        ResizableVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~ResizableVectorDecorator() override {}

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        virtual void setNumElements(uint32 numElements, bool freeMemory) {
            Vector::view.resize(numElements, freeMemory);
        }
};

/**
 * Allows to set all values stored in a vector to zero.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class ClearableVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        ClearableVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~ClearableVectorDecorator() override {}

        /**
         * Sets all values stored in the vector to zero.
         */
        virtual void clear() {
            setViewToZeros(Vector::view.array, Vector::view.numElements);
        }
};
