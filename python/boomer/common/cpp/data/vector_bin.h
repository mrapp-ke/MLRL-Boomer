/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "indexed_value.h"
#include "bin.h"
#include "vector_dense.h"
#include <unordered_map>
#include <forward_list>


/**
 * An one-dimensional vector that provides random access to a fixed number of bins stored in a C-contiguous array, as
 * well as to the indices and feature values of the examples that belong to the individual bins.
 */
class BinVector final : public DenseVector<Bin> {

    public:

        typedef IndexedValue<float32> Example;

        typedef std::forward_list<Example> ExampleList;

    private:

        std::unordered_map<uint32, ExampleList> examplesPerBin_;

    public:

        /**
         * @param numElements The number of bins in the vector
         */
        BinVector(uint32 numElements);

        /**
         * @param numElements   The number of bins in the vector
         * @param init          True, if all bins in the vector should be value-initialized, false otherwise
         */
        BinVector(uint32 numElements, bool init);

        /**
         * Returns the list that stores the examples that correspond to a specific bin.
         *
         * @param binIndex  The index of the bin
         * @return          A reference to an object of type `ExampleList` that stores the examples in the bin
         */
        ExampleList& getExamples(uint32 binIndex);

};
