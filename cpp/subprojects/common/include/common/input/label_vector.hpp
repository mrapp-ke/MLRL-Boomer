/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * An one-dimensional sparse vector that stores the indices of labels that are relevant to an example.
 */
class LabelVector final {

    private:

        uint32 numElements_;

        uint32 maxCapacity_;

        uint32* array_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        LabelVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all values in the vector should be value-initialized, false otherwise
         */
        LabelVector(uint32 numElements, bool init);

        /**
         * An iterator that provides access to the indices in the vector and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * An iterator that provides read-only access to the indices in the vector.
         */
        typedef const uint32* index_const_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the indices in the vector.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin();

        /**
         * Returns an `index_iterator` to the end of the indices in the vector.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the indices in the vector.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices in the vector.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);

};
