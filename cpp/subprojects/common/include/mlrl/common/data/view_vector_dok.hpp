/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <unordered_map>

/**
 * A view that provides access to values stored in a sparse vector in the dictionary of keys (DOK) format.
 *
 * @tparam T The type of the valeus stored in the vector
 */
template<typename T>
class DokVector {
    protected:

        /**
         * A pointer to an object of type `std::unordered_map` that stores the indices and values of all non-zero
         * elements in the view.
         */
        std::unordered_map<uint32, T>* values_;

    public:

        /**
         * The value of sparse elements.
         */
        const T sparseValue;

        /**
         * @param values        A pointer to an object of type `std::unordered_map` that stores the indices and values
         *                      of all non-zero elements in the view
         * @param sparseValue   The value of sparse elements
         */
        DokVector(std::unordered_map<uint32, T>* values, T sparseValue) : values_(values), sparseValue(sparseValue) {}

        /**
         * @param other A reference to an object of type `DokVector` that should be copied
         */
        DokVector(const DokVector& other) : values_(other.values_), sparseValue(other.sparseValue) {}

        /**
         * @param other A reference to an object of type `DokVector` that should be moved
         */
        DokVector(DokVector&& other) : values_(other.values_), sparseValue(other.sparseValue) {}

        virtual ~DokVector() {}

        /**
         * The type of the indices, the view provides access to.
         */
        typedef uint32 index_type;

        /**
         * The type of the values, the view provides access to.
         */
        typedef T value_type;

        /**
         * An iterator that provides read-only access to non-zero elements in the vector.
         */
        typedef typename std::unordered_map<uint32, T>::const_iterator const_iterator;

        /**
         * An iterator that provides access to non-zero elements in the vector and allows to modify them.
         */
        typedef typename std::unordered_map<uint32, T>::iterator iterator;

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return values_->cbegin();
        }

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const {
            return values_->cend();
        }

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin() {
            return values_->begin();
        }

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end() {
            return values_->end();
        }

        /**
         * Returns the value of the element at a specific index.
         *
         * @param index The index of the element
         * @return      The value of the element at the given index
         */
        const value_type& operator[](index_type index) const {
            auto it = values_->find(index);
            return it != values_->cend() ? it->second : sparseValue;
        }

        /**
         * Sets the value of the element at a specific position.
         *
         * @param index The index of the element
         * @param value The value to be set
         */
        void set(index_type index, value_type value) {
            auto result = values_->emplace(index, value);

            if (!result.second) {
                result.first->second = value;
            }
        }

        /**
         * Sets all values stored in the view to zero.
         */
        void clear() {
            values_->clear();
        }
};

/**
 * Allocates the memory for a view that provides access to values stored in a sparse vector in the dictionary of keys
 * (DOK) format.
 *
 * @tparam Vector The type of the view
 */
template<typename Vector>
class DokVectorAllocator : public Vector {
    public:

        /**
         * @param sparseValue The value of sparse elements
         */
        DokVectorAllocator(typename Vector::value_type sparseValue = 0)
            : Vector(new std::unordered_map<typename Vector::index_type, typename Vector::value_type>(), sparseValue) {}

        /**
         * @param other A reference to an object of type `DokVectorAllocator` that should be copied
         */
        DokVectorAllocator(const DokVectorAllocator& other) : Vector(other) {
            throw std::runtime_error("Objects of type DokVectorAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `DokVectorAllocator` that should be moved
         */
        DokVectorAllocator(DokVectorAllocator&& other) : Vector(std::move(other)) {
            other.values_ = nullptr;
        }

        virtual ~DokVectorAllocator() override {
            delete Vector::values_;
        }
};

/**
 * Allocates the memory, a `DokVector` provides access to.
 *
 * @tparam T The type of the values stored in the `DokVector`
 */
template<typename T>
using AllocatedDokVector = DokVectorAllocator<DokVector<T>>;

/**
 * Provides read and write access via iterators to all non-zero values stored in a sparse vector in the dictionary of
 * keys (DOK) format.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class IterableDokVectorDecorator : public Vector {
    public:

        /**
         * An iterator that provides read-only access to non-zero elements in the vector.
         */
        typedef typename Vector::view_type::const_iterator const_iterator;

        /**
         * An iterator that provides access to non-zero elements in the vector and allows to modify them.
         */
        typedef typename Vector::view_type::iterator iterator;

        /**
         * @param view The view, the vector should be backed by
         */
        IterableDokVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~IterableDokVectorDecorator() override {}

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return Vector::view.cbegin();
        }

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const {
            return Vector::view.cend();
        }

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin() {
            return Vector::view.begin();
        }

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end() {
            return Vector::view.end();
        }
};
