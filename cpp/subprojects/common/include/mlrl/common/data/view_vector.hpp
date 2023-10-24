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
struct OneDimensionalView : public View<T> {
        /**
         * The number of elements in the view.
         */
        uint32 numElements;

        /**
         * @param a A pointer to an array of template type `T` that stores the values, the view should provide access to
         * @param n The number of elements in the view
         */
        OneDimensionalView(T* a, uint32 n) : View<T>(a), numElements(n) {}

        /**
         * @param other A const reference to an object of type `OneDimensionalView` that should be copied
         */
        OneDimensionalView(const OneDimensionalView<T>& other) : OneDimensionalView(other.array, other.numElements) {}

        /**
         * @param other A reference to an object of type `OneDimensionalView` that should be moved
         */
        OneDimensionalView(OneDimensionalView<T>&& other) : OneDimensionalView(other.array, other.numElements) {}

        virtual ~OneDimensionalView() override {};
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
 * Provides random read-only access to the values stored in a vector.
 *
 * @tparam Base The type of the vector
 */
template<typename Vector>
class ReadAccessibleVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        ReadAccessibleVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~ReadAccessibleVectorDecorator() override {};

        /**
         * Returns a const reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A const reference to the specified element
         */
        const typename Vector::value_type& operator[](uint32 pos) const {
            return Vector::view_.array[pos];
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
 * Provides random write access to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class WriteAccessibleVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        WriteAccessibleVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~WriteAccessibleVectorDecorator() override {};

        /**
         * Returns a const reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A const reference to the specified element
         */
        const typename Vector::value_type& operator[](uint32 pos) const {
            return Vector::view_.array[pos];
        }

        /**
         * Returns a reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A reference to the specified element
         */
        typename Vector::value_type& operator[](uint32 pos) {
            return Vector::view_.array[pos];
        }
};

/**
 * Provides random read and write access to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
using AccessibleVectorDecorator = WriteAccessibleVectorDecorator<ReadAccessibleVectorDecorator<Vector>>;

/**
 * Implements read and write access to the values that are stored in a pre-allocated C-contiguous array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class MLRLCOMMON_API VectorView : public VectorConstView<T> {
    public:

        /**
         * @param numElements   The number of elements in the view
         * @param array         A pointer to a C-contiguous array of template type `T` that stores the values, the view
         *                      provides access to
         */
        VectorView(uint32 numElements, T* array);

        virtual ~VectorView() override {};

        // Keep functions from the parent class rather than hiding them
        using VectorConstView<T>::operator[];

        /**
         * An iterator that provides access to the elements in the view and allows to modify them.
         */
        typedef T* iterator;

        /**
         * Returns an `iterator` to the beginning of the view.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the view.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a reference to the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      A reference to the specified element
         */
        T& operator[](uint32 pos);
};
