/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/indexed_value.h"
#include "../data/vector_dense.h"
#include "../data/bin.h"
#include <unordered_map>
#include <forward_list>


/**
 * An one-dimensional vector that stores a fixed number of bins, as well as the indices and feature values of the
 * examples that belong to the individual bins.
 */
class BinVector final {

    public:

        typedef IndexedValue<float32> Example;

        typedef std::forward_list<Example> ExampleList;

    private:

        DenseVector<Bin> vector_;

        std::unordered_map<uint32, ExampleList> examplesPerBin_;

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

        typedef DenseVector<Bin>::iterator bin_iterator;

        typedef DenseVector<Bin>::const_iterator bin_const_iterator;

        /**
         * Returns a `bin_iterator` to the beginning of the bins.
         *
         * @return A `bin_iterator` to the beginning
         */
        bin_iterator bins_begin();

        /**
         * Returns an `bin_iterator` to the end of the bins.
         *
         * @return An `bin_iterator` to the end
         */
        bin_iterator bins_end();

        /**
         * Returns a `bin_const_iterator` to the beginning of the bins.
         *
         * @return A `bin_const_iterator` to the beginning
         */
        bin_const_iterator bins_cbegin() const;

        /**
         * Returns a `bin_const_iterator` to the end of the bins.
         *
         * @return A `bin_const_iterator` to the end
         */
        bin_const_iterator bins_cend() const;

        /**
         * Returns the list that stores the examples that correspond to a specific bin.
         *
         * @param binIndex  The index of the bin
         * @return          A reference to an object of type `ExampleList` that stores the examples in the bin
         */
        ExampleList& getExamples(uint32 binIndex);

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


        /**
         * Swaps the examples that correspond to the bin at a specific position with the examples of another bin.
         *
         * @param pos1  The position of the first bin
         * @param pos2  The position of the secon bin
         */
        void swapExamples(uint32 pos1, uint32 pos2);

};
