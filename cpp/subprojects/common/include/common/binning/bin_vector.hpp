/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include "common/data/vector_dense.hpp"
#include "common/data/vector_mapping_dense.hpp"
#include "common/data/bin.hpp"


class BinVectorNew final : DenseVector<Bin> {

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        BinVectorNew(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        BinVectorNew(uint32 numElements, bool init);

};

/**
 * An one-dimensional vector that stores a fixed number of bins, as well as the indices and feature values of the
 * examples that belong to the individual bins.
 */
class BinVector final {

    public:

        typedef IndexedValue<float32> Example;

    private:

        DenseVector<Bin> vector_;

        DenseMappingVector<Example> mapping_;

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

        typedef DenseMappingVector<Example>::iterator example_list_iterator;

        typedef DenseMappingVector<Example>::const_iterator example_list_const_iterator;

        typedef DenseMappingVector<Example>::Entry ExampleList;

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
         * Returns an `example_list_iterator` to the beginning of the examples.
         * Returns the list that stores the examples that correspond to a specific bin.
         *
         * @return An `example_list_iterator` to the beginning
         */
        example_list_iterator examples_begin();

        /**
         * Returns an `example_list_iterator` to the end of the examples.
         *
         * @return An `example_list_iterator` to the end
         */
        example_list_iterator examples_end();

        /**
         * Returns an `example_list_const_iterator` to the beginning of the examples.
         *
         * @return An `example_list_const_iterator` to the beginning
         */
        example_list_const_iterator examples_cbegin() const;

        /**
         * Returns an `example_list_const_iterator` to the end of the examples.
         *
         * @return An `example_list_const_iterator` to the end
         */
        example_list_const_iterator examples_cend() const;

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
