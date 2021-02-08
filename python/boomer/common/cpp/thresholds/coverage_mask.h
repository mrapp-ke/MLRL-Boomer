/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/types.h"


/**
 * Allows to keep track of the elements, e.g. examples or bins, that are covered by a rule as the rule is refined. For
 * each element, an integer is stored in a C-contiguous array that may be updated when the rule is refined. The elements
 * that correspond to a number that is equal to the "target" are considered to be covered.
 */
class CoverageMask final {

    private:

        uint32* array_;

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of elements
         */
        CoverageMask(uint32 numElements);

        /**
         * @param coverageMask A reference to an object of type `CoverageMask` to be copied
         */
        CoverageMask(const CoverageMask& coverageMask);

        ~CoverageMask();

        typedef uint32* iterator;

        /**
         * Returns an `iterator` to the beginning of the mask.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the mask.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Resets the mask by setting all elements and the "target" to zero.
         */
        void reset();

        /**
         * Returns whether the element at a specific element it covered or not.
         *
         * @param pos   The position of the element to be checked
         * @return      True, if the element at the given position is covered, false otherwise
         */
        bool isCovered(uint32 pos) const;

        /**
         * The "target" that corresponds to the elements that are considered to be covered.
         */
        uint32 target;

        /**
         * Returns the number of elements in the coverage mask.
         *
         * @return the number of elements
         */
        uint32 getNumElements() const;
};
