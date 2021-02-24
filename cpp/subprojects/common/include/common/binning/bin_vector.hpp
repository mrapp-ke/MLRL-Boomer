/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_dense.hpp"
#include "common/data/bin.hpp"


/**
 * An one-dimensional vector that stores a fixed number of bins.
 */
class BinVector final : public DenseVector<Bin> {

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        BinVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        BinVector(uint32 numElements, bool init);

};
