/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/view_vector.hpp"

#include <limits>
#include <vector>

/**
 * An one-dimensional view that provides random read and write access, as well as read and write access via iterators,
 * to sparse elements, consisting of an index and a value, stored in a dynamic array. Compared to the view
 * `SparseArrayVector`, the ability to provide random access to the elements in the view comes at the expense of memory
 * efficiency, as it does not only require to store the sparse elements, but also a fixed-size array that stores for
 * each position the index of the corresponding element in the dynamic array, if available.
 *
 * This data structure is often referred to as an "unordered sparse set". It was originally proposed in "An efficient
 * representation for sparse sets", Briggs, Torczon, 1993 (see
 * https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c83ae968c51219db68a03fc5b01de751dd2fe9ae).
 *
 * @tparam T            The type of the values, the view provides access to
 * @tparam ValueArray   The type of the dynamic array that stores the sparse elements
 * @tparam IndexType    the type of the fixed-size array that stores for each position the index of the corresponding
 *                      element in the dynamic array
 */
template<typename T, typename ValueArray = std::vector<IndexedValue<T>>, typename IndexType = uint32>
class SparseSetVector {
    public:

        /**
         * The index that is used to indicate that the value of a specific elements is zero.
         */
        static inline constexpr uint32 MAX_INDEX = std::numeric_limits<IndexType>::max();

        /**
         * A pointer to the dynamic array that stores the sparse elements.
         */
        ValueArray* values;

        /**
         * A pointer to the fixed-size array that stores for each position the index of the corresponding element in the
         * dynamic array, if available.
         */
        IndexType* indices;

        /**
         * The number of elements in the view.
         */
        uint32 numElements;

        /**
         * @param values        A pointer to an object of template type `ValueArray` that stores the sparse elements
         * @param indices       A pointer to an array of template type `IndexType` that stores for each position the
         *                      index of the corresponding element in `values`
         * @param numElements   The number of elements in the view
         */
        explicit SparseSetVector(ValueArray* values, IndexType* indices, uint32 numElements)
            : values(values), indices(indices), numElements(numElements) {}

        /**
         * @param other A reference to an object of type `SparseSetVector` that should be copied
         */
        SparseSetVector(const SparseSetVector<T, ValueArray, IndexType>& other)
            : SparseSetVector<T, ValueArray, IndexType>(other.values, other.indices, other.numElements) {}

        /**
         * @param other A reference to an object of type `SparseSetVector` that should be moved
         */
        SparseSetVector(SparseSetVector<T, ValueArray, IndexType>&& other)
            : SparseSetVector<T, ValueArray, IndexType>(other.values, other.indices, other.numElements) {}

        virtual ~SparseSetVector() {}

        /**
         * The type of the values in the view.
         */
        typedef IndexedValue<T> value_type;

        /**
         * An iterator that provides read-only access to the sparse elements in the view.
         */
        typedef typename ValueArray::const_iterator const_iterator;

        /**
         * An iterator that provides access to the sparse elements in the row and allows to modify them.
         */
        typedef typename ValueArray::iterator iterator;

        /**
         * Returns a `const_iterator` to the beginning of the view.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return values->cbegin();
        }

        /**
         * Returns a `const_iterator` to the end of the view.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const {
            return values->cend();
        }

        /**
         * Returns an `iterator` to the beginning of the view.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin() {
            return values->begin();
        }

        /**
         * Returns an `iterator` to the end of the view.
         *
         * @return An `iterator` to the end
         */
        iterator end() {
            return values->end();
        }

        /**
         * Returns a pointer to the element that corresponds to a specific index.
         *
         * @param index The index of the element to be returned
         * @return      A pointer to the element that corresponds to the given index or a null pointer, if no such
         *              element is available
         */
        const value_type* operator[](uint32 index) const {
            IndexType i = indices[index];
            return i == MAX_INDEX ? nullptr : &((*values)[i]);
        }

        /**
         * Returns a pointer to the element that corresponds to a specific index.
         *
         * @param index The index of the element to be returned
         * @return      A pointer to the element that corresponds to the given index or a null pointer, if no such
         *              element is available
         */
        value_type* operator[](uint32 index) {
            IndexType i = indices[index];
            return i == MAX_INDEX ? nullptr : &((*values)[i]);
        }

        /**
         * Returns a reference to the element that corresponds to a specific index. If no such element is available, it
         * is inserted into the vector.
         *
         * @param index The index of the element to be returned or inserted
         * @return      A reference to the element that corresponds to the given index
         */
        value_type& emplace(uint32 index) {
            IndexType i = indices[index];

            if (i == MAX_INDEX) {
                indices[index] = static_cast<IndexType>(values->size());
                values->emplace_back(index);
                return values->back();
            }

            return (*values)[i];
        }

        /**
         * Returns a reference to the element that corresponds to a specific index. If no such element is available, it
         * is inserted into the vector using a specific default value.
         *
         * @param index         The index of the element to be returned or inserted
         * @param defaultValue  The default value to be used if a new element is inserted
         * @return              A reference to the element that corresponds to the given index
         */
        value_type& emplace(uint32 index, const T& defaultValue) {
            IndexType i = indices[index];

            if (i == MAX_INDEX) {
                indices[index] = static_cast<IndexType>(values->size());
                values->emplace_back(index, defaultValue);
                return values->back();
            }

            return (*values)[i];
        }

        /**
         * Removes the element that corresponds to a specific index, if available.
         *
         * @param index The index of the element to be removed
         */
        void erase(uint32 index) {
            IndexType i = indices[index];

            if (i != MAX_INDEX) {
                const value_type& lastEntry = values->back();
                uint32 lastIndex = lastEntry.index;

                if (lastIndex != index) {
                    (*values)[i] = lastEntry;
                    indices[lastIndex] = i;
                }

                indices[index] = MAX_INDEX;
                values->pop_back();
            }
        }

        /**
         * Removes all elements from the row.
         */
        void clear() {
            while (!values->empty()) {
                const value_type& lastEntry = values->back();
                indices[lastEntry.index] = MAX_INDEX;
                values->pop_back();
            }
        }

        /**
         * Releases the ownership of the dynamic array that stores the sparse elements in the view. As a result, the
         * behavior of this view becomes undefined and it should not be used anymore. The caller is responsible for
         * freeing the memory that is occupied by the array.
         *
         * @return A pointer to an object of template type `ValueArray` that stores the sparse elements in the view
         */
        ValueArray* releaseValues() {
            ValueArray* ptr = values;
            values = nullptr;
            return ptr;
        }

        /**
         * Releases the ownership of the fixed-size array that stores for each position the index of the corresponding
         * element in the dynamic array. As a result, the behavior of this view becomes undefined and it should not be
         * used anymore. The caller is responsible for freeing the memory that is occupied by the array.
         *
         * @return A pointer to an array of template type `IndexType` that stores for each position the index of the
         *         corresponding element in the dynamic array
         */
        IndexType* releaseIndices() {
            IndexType* ptr = indices;
            indices = nullptr;
            return ptr;
        }
};
