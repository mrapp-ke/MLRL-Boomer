/**
 * Provides interfaces and classes that provide access to indices that allow to restrict the access to data that is
 * stored in matrices or vectors to a subset of the available data.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"


/**
 * Defines an interface for all one-dimensional vectors that provide random access to indices.
 */
class IIndexVector : virtual public IRandomAccessVector<uint32> {

    public:

        virtual ~IIndexVector() { };

};

/**
 * An one-dimensional vector that provides random access to a fixed number of indices stored in a C-contiguous array.
 */
class DenseIndexVector : public DenseVector<uint32>, virtual public IIndexVector {

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseIndexVector(uint32 numElements);

};

/**
 * An one-dimensional vector that provides random access to all indices within a continuous range [0, numIndices).
 */
class RangeIndexVector : virtual public IIndexVector {

    private:

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of indices, the vector provides access to
         */
        RangeIndexVector(uint32 numElements);

        uint32 getNumElements() const override;

        uint32 getValue(uint32 pos) const override;

};
