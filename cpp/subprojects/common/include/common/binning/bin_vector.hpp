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

};
