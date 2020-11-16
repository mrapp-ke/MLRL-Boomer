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
class BinVector : public DenseVector<Bin> {

    public:

        typedef IndexedValue<float32> Example;

    private:

        std::unordered_map<uint32, std::forward_list<Example>> examplesPerBin_;

    public:

        /**
         * @param numElements The number of bins in the vector
         */
        BinVector(uint32 numElements);

        typedef std::forward_list<Example>::const_iterator example_const_iterator;

        /**
         * Returns an `example_const_iterator` to the beginning of the examples in a certain bin.
         *
         * @param binIndex  The index of the bin
         * @return          An `example_const_iterator` to the beginning
         */
        example_const_iterator examples_cbegin(uint32 binIndex);

        /**
         * Returns an `example_const_iterator` to the end of the examples in a certain bin.
         *
         * @param binIndex  The index of the bin
         * @return          An `example_const_iterator` to the end
         */
        example_const_iterator examples_cend(uint32 binIndex);

        /**
         * Adds a new example to a certain bin.
         *
         * @param binIndex  The index of the bin
         * @param example   The example to be added
         */
        void addExample(uint32 binIndex, Example example);

        /**
         * Removes all examples, regardless of the bins they belong to.
         */
        void clearAllExamples();

};
