/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"

#include <utility>

/**
 * A one-dimensional view that provides access to indices and corresponding values stored in pre-allocated arrays of the
 * same size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API IndexedVectorView {
    protected:

        /**
         * A view that provides access to the indices.
         */
        Vector<uint32> indices_;

        /**
         * A view that provides access to the values.
         */
        View<T> values_;

    public:

        /**
         * @param indices       A pointer to an array of type `uint32` that stores the values, the view should provide
         *                      access to
         * @param values        A pointer to an array of template type `T` that stores the values, the view should
         *                      provide access to
         * @param numElements   The number of elements in the view
         */
        explicit IndexedVectorView(uint32* indices, T* values, uint32 numElements)
            : indices_(indices, numElements), values_(values) {}

        /**
         * @param other A const reference to an object of type `Vector` that should be copied
         */
        IndexedVectorView(const IndexedVectorView<T>& other) : indices_(other.indices_), values_(other.values_) {}

        /**
         * @param other A reference to an object of type `Vector` that should be moved
         */
        IndexedVectorView(IndexedVectorView<T>&& other)
            : indices_(std::move(other.indices_)), values_(std::move(other.values_)) {}

        virtual ~IndexedVectorView() {}

        /**
         * The type of the indices that are stored in the vector.
         */
        using index_type = uint32;

        /**
         * The type of the values that are stored in the vector.
         */
        using value_type = T;

        /**
         * An iterator that provides read-only access to the indices stored in the vector.
         */
        using index_const_iterator = View<uint32>::const_iterator;

        /**
         * An iterator that provides read-only access to the values stored in the vector.
         */
        using value_const_iterator = typename Vector<T>::const_iterator;

        /**
         * An iterator that provides access to the indices stored in the vector and allows to modify them.
         */
        using index_iterator = View<uint32>::iterator;

        /**
         * An iterator that provides access to the values stored in the vector and allows to modify them.
         */
        using value_iterator = typename Vector<T>::iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the vector.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const {
            return indices_.cbegin();
        }

        /**
         * Returns an `index_const_iterator` to the end of the vector.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const {
            return indices_.cend();
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the vector.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const {
            return values_.cbegin();
        }

        /**
         * Returns a `value_const_iterator` to the end of the vector.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const {
            return &values_.array[indices_.numElements];
        }

        /**
         * Returns an `index_iterator` to the beginning of the vector.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin() {
            return indices_.begin();
        }

        /**
         * Returns an `index_iterator` to the end of the vector.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end() {
            return indices_.end();
        }

        /**
         * Returns a `value_iterator` to the beginning of the vector.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin() {
            return values_.begin();
        }

        /**
         * Returns a `value_iterator` to the end of the vector.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end() {
            return &values_.array[indices_.numElements];
        }

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const {
            return indices_.numElements;
        }
};

/**
 * Allocates the memory, a `IndexedVectorView` provides access to.
 *
 * @tparam View             The type of the view
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename View, typename MemoryAllocator = DefaultMemoryAllocator>
class MLRLCOMMON_API IndexedVectorAllocator : public View {
    public:

        /**
         * @param numElements   The number of elements in the view
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit IndexedVectorAllocator(uint32 numElements, bool init = false)
            : View(MemoryAllocator::template allocateMemory<typename View::index_type>(numElements, init),
                   MemoryAllocator::template allocateMemory<typename View::value_type>(numElements, init),
                   numElements) {}

        /**
         * @param other A reference to an object of type `IndexedVectorAllocator` that should be copied
         */
        IndexedVectorAllocator(const IndexedVectorAllocator<View, MemoryAllocator>& other) : View(other) {
            throw std::runtime_error("Objects of type IndexedVectorAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `IndexedVectorAllocator` that should be moved
         */
        IndexedVectorAllocator(IndexedVectorAllocator<View, MemoryAllocator>&& other) : View(std::move(other)) {
            other.indices_.release();
            other.values_.release();
        }

        virtual ~IndexedVectorAllocator() override {
            MemoryAllocator::freeMemory(View::indices_.array);
            MemoryAllocator::freeMemory(View::values_.array);
        }
};

/**
 * Allocates the memory, a `IndexedVectorView` provides access to.
 *
 * @tparam T The type of the values stored in the `IndexedVectorView`
 */
template<typename T>
using AllocatedIndexedVector = IndexedVectorAllocator<IndexedVectorView<T>>;

/**
 * A vector that is backed by two one-dimensional views of a specific size, storing indices and corresponding values.
 *
 * @tparam Vector The type of the view, the vector is backed by
 */
template<typename Vector>
class MLRLCOMMON_API IndexedVectorDecorator : public ViewDecorator<Vector> {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit IndexedVectorDecorator(Vector&& view) : ViewDecorator<Vector>(std::move(view)) {}

        virtual ~IndexedVectorDecorator() override {}

        /**
         * The type of the indices that are stored in the vector.
         */
        using index_type = typename Vector::value_type;

        /**
         * The type of the values that are stored in the vector.
         */
        using value_type = typename Vector::value_type;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const {
            return this->view.getNumElements();
        }
};

/**
 * Provides access via iterators to indices and corresponding values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class MLRLCOMMON_API IterableIndexedVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit IterableIndexedVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~IterableIndexedVectorDecorator() override {}

        /**
         * An iterator that provides read-only access to the indices stored in the vector.
         */
        using index_const_iterator = typename Vector::view_type::index_const_iterator;

        /**
         * An iterator that provides read-only access to the values stored in the vector.
         */
        using value_const_iterator = typename Vector::view_type::value_const_iterator;

        /**
         * An iterator that provides access to the indices stored in the vector and allows to modify them.
         */
        using index_iterator = typename Vector::view_type::index_iterator;

        /**
         * An iterator that provides access to the values stored in the vector and allows to modify them.
         */
        using value_iterator = typename Vector::view_type::value_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the vector.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const {
            return Vector::view.indices_cbegin();
        }

        /**
         * Returns an `index_const_iterator` to the end of the vector.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const {
            return Vector::view.indices_cend();
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the vector.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const {
            return Vector::view.values_cbegin();
        }

        /**
         * Returns a `value_const_iterator` to the end of the vector.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const {
            return Vector::view.values_cend();
        }

        /**
         * Returns an `index_iterator` to the beginning of the vector.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin() {
            return Vector::view.indices_begin();
        }

        /**
         * Returns an `index_iterator` to the end of the vector.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end() {
            return Vector::view.indices_end();
        }

        /**
         * Returns a `value_iterator` to the beginning of the vector.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin() {
            return Vector::view.values_begin();
        }

        /**
         * Returns a `value_iterator` to the end of the vector.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end() {
            return Vector::view.values_end();
        }
};
