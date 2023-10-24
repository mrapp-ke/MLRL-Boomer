/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"
#include "mlrl/common/data/view_one_dimensional.hpp"

/**
 * A one-dimensional view that provides access to values stored in a pre-allocated array of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
struct Vector : public View<T> {
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

        virtual ~Vector() override {};
};

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

        virtual ~VectorDecorator() override {};

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const {
            return ViewDecorator<View>::view_.numElements;
        }
};

/**
 * Implements read-only access to the values that are stored in a pre-allocated C-contiguous array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class MLRLCOMMON_API VectorConstView : public IOneDimensionalView {
    protected:

        /**
         * The number of elements in the view.
         */
        uint32 numElements_;

        /**
         * A pointer to the array that stores the values, the view provides access to.
         */
        T* array_;

    public:

        /**
         * @param numElements   The number of elements in the view
         * @param array         A pointer to a C-contiguous array of template type `T` that stores the values, the view
         *                      provides access to
         */
        VectorConstView(uint32 numElements, T* array);

        virtual ~VectorConstView() override {};

        /**
         * An iterator that provides read-only access to the elements in the view.
         */
        typedef const T* const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the view.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the view.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns a const reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A const reference to the specified element
         */
        const T& operator[](uint32 pos) const;

        /**
         * @see `IOneDimensionalView::getNumElements`
         */
        uint32 getNumElements() const override final;
};

/**
 * Provides random read and write access to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
using AccessibleVectorDecorator = WriteAccessibleViewDecorator<ReadAccessibleViewDecorator<VectorDecorator<Vector>>>;

/**
 * Provides read-only access via iterators to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class ReadIterableVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        ReadIterableVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~ReadIterableVectorDecorator() override {};

        /**
         * An iterator that provides read-only access to the values stored in the vector.
         */
        typedef typename Vector::view_type::const_iterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return Vector::view_.array;
        }

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const {
            return &Vector::view_.array[Vector::view_.numElements];
        }
};

/**
 * Provides write access via iterators to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class WriteIterableVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        WriteIterableVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~WriteIterableVectorDecorator() override {};

        /**
         * An iterator that provides access to the values stored in the vector and allows to modify them.
         */
        typedef typename Vector::view_type::iterator iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin() {
            return Vector::view_.array;
        }

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end() {
            return &Vector::view_.array[Vector::view_.numElements];
        }
};

/**
 * Provides read and write access via iterators to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
using IterableVectorDecorator = WriteIterableVectorDecorator<ReadIterableVectorDecorator<Vector>>;

/**
 * Provides random read-only access, as well as read-only access via iterators, to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
using ReadableVectorDecorator = ReadAccessibleViewDecorator<ReadIterableVectorDecorator<VectorDecorator<Vector>>>;

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values stored in a
 * vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
using WritableVectorDecorator =
  WriteIterableVectorDecorator<WriteAccessibleViewDecorator<ReadableVectorDecorator<Vector>>>;
