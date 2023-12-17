/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <unordered_map>

/**
 * An one-dimensional sparse vector that stores data using the dictionary of keys (DOK) format.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<typename T>
class DokVector final {
    private:

        std::unordered_map<uint32, T> data_;

        const T sparseValue_;

    public:

        /**
         * @param sparseValue The value of sparse elements
         */
        DokVector(T sparseValue) : sparseValue_(sparseValue) {}

        /**
         * An iterator that provides access to the elements in the vector and allows to modify them.
         */
        typedef typename std::unordered_map<uint32, T>::iterator iterator;

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef typename std::unordered_map<uint32, T>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin() {
            return data_.begin();
        }

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end() {
            return data_.end();
        }

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return data_.cbegin();
        }

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const {
            return data_.cend();
        }

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      The value of the specified element
         */
        const T& operator[](uint32 pos) const {
            auto it = data_.find(pos);
            return it != data_.cend() ? it->second : sparseValue_;
        }

        /**
         * Sets a value to the element at a specific position.
         *
         * @param pos   The position of the element
         * @param value The value to be set
         */
        void set(uint32 pos, T value) {
            auto result = data_.emplace(pos, value);

            if (!result.second) {
                result.first->second = value;
            }
        }

        /**
         * Sets the values of all elements to zero.
         */
        void clear() {
            data_.clear();
        }
};
