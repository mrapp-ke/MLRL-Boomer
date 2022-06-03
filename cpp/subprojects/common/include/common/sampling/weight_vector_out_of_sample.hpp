/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

/**
 * An one-dimensional vector that provides random access to a fixed number of weights that are obtained from another
 * vector by setting zero weights to one and non-zero weights to zero.
 *
 * @tparam T The type of the other vector
 */
template<typename T>
class OutOfSampleWeightVector final {

    private:

        const T& vector_;

    public:

        /**
         * @param vector A reference to an object of template type `T` that provides access to the original weights
         */
        OutOfSampleWeightVector(const T& vector);

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements
         */
        uint32 getNumElements() const;

        /**
         * Returns the weight at a specific position.
         *
         * @param pos   The position
         * @return      The weight at the specified position
         */
        bool operator[](uint32 pos) const;

};
