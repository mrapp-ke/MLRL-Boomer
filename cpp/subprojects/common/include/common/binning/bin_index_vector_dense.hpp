/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/binning/bin_index_vector.hpp"
#include "common/data/vector_dense.hpp"


/**
 * Stores the indices of the bins, individual examples have been assigned to, using a C-contiguous array.
 */
class DenseBinIndexVector final : public IBinIndexVector {

    private:

        DenseVector<uint32> vector_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseBinIndexVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseBinIndexVector(uint32 numElements, bool init);

        typedef DenseVector<uint32>::iterator iterator;

        typedef DenseVector<uint32>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the indices.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the indices.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the indices.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the indices.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

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

        uint32 getBinIndex(uint32 exampleIndex) const override;

};
