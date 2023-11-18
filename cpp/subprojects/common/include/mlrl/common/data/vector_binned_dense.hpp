/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/array.hpp"
#include "mlrl/common/data/view_vector_binned.hpp"
#include "mlrl/common/iterator/binned_iterator.hpp"

/**
 * An one-dimensional vector that provides random access to a fixed number of elements, corresponding to bins, stored in
 * a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<typename T>
class DenseBinnedVector final : public WritableBinnedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<T>> {
    private:

        uint32 maxCapacity_;

    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param numBins       The number of bins
         */
        DenseBinnedVector(uint32 numElements, uint32 numBins);

        virtual ~DenseBinnedVector() override {};

        /**
         * Sets the number of bins.
         *
         * @param numBins       The number of bins to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumBins(uint32 numBins, bool freeMemory);
};
