/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"

#include <utility>

/**
 * A ring buffer with fixed capacity.
 *
 * @tparam T The type of the values that are stored in the buffer
 */
template<typename T>
class RingBuffer final : ViewDecorator<AllocatedVector<T>> {
    private:

        uint32 pos_;

        bool full_;

    public:

        /**
         * @param capacity The maximum capacity of the buffer. Must be at least 1
         */
        RingBuffer(uint32 capacity)
            : ViewDecorator<AllocatedVector<T>>(AllocatedVector<T>(capacity)), pos_(0), full_(capacity == 0) {}

        /**
         * An iterator that provides read-only access to the elements in the buffer.
         */
        using const_iterator = View<T>::const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the buffer.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return this->view.cbegin();
        }

        /**
         * Returns a `const_iterator` to the end of the buffer.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const {
            return &this->view.array[full_ ? this->getCapacity() : pos_];
        }

        /**
         * Returns the maximum capacity of the buffer.
         *
         * @return The maximum capacity
         */
        uint32 getCapacity() const {
            return this->view.numElements;
        }

        /**
         * Returns the number of values in the buffer.
         *
         * @return The number of values
         */
        uint32 getNumElements() const {
            return full_ ? this->getCapacity() : pos_;
        }

        /**
         * Returns whether the maximum capacity of the buffer has been reached or not.
         *
         * @return True, if the maximum capacity has been reached, false otherwise
         */
        bool isFull() const {
            return full_;
        }

        /**
         * Adds a new value to the buffer. If the maximum capacity of the buffer has been reached, the oldest value will
         * be overwritten.
         *
         * @param value The value to be added
         * @return      A `std::pair`, whose first value indicates whether a value has been overwritten or not. If a
         *              value has been overwritten, the pair's second value is set to the overwritten value, otherwise
         *              it is undefined
         */
        std::pair<bool, T> push(T value) {
            std::pair<bool, T> result;
            result.first = full_;
            result.second = this->view.array[pos_];
            this->view.array[pos_] = value;
            pos_++;

            if (pos_ >= this->getCapacity()) {
                pos_ = 0;
                full_ = true;
            }

            return result;
        }
};
