/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <unordered_map>


/**
 * An one-dimensional sparse vector that stores data using the dictionary of keys (DOK) format.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<class T>
class DokVector final {

    private:

        std::unordered_map<uint32, T> data_;

        T sparseValue_;

    public:

        /**
         * @param sparseValue The value of sparse elements
         */
        DokVector(T sparseValue);

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @return      The value of the given element
         */
        T getValue(uint32 pos) const;

        /**
         * Sets a value to the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @param value The value to be set
         */
        void setValue(uint32 pos, T value);

        /**
         * Sets the values of all elements to zero.
         */
        void setAllToZero();

};
