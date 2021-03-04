/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Defines an interface for all classes that provide access to the indices of the bins, individual examples have been
 * assigned to.
 */
class IBinIndexVector {

    public:

        virtual ~IBinIndexVector() { };

        /**
         * Returns the index of the bin, the example at a specific index has been assigned to.
         *
         * @param exampleIndex  The index of the example
         * @return              The index of the bin, the example has been assigned to
         */
        virtual uint32 getBinIndex(uint32 exampleIndex) const = 0;

        /**
         * Sets the index of the bin, the examples at a specific index should be assigned to.
         *
         * @param exampleIndex  The index of the example
         * @param binIndex      The index of the bin, the example should be assigned to
         */
        virtual void setBinIndex(uint32 exampleIndex, uint32 binIndex) = 0;

};
