/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_binned.hpp"

#include <utility>

/**
 * Allocates the memory, a `BinnedFeatureVector` provides access to.
 */
class MLRLCOMMON_API AllocatedBinnedFeatureVector : public BinnedFeatureVector {
    public:

        /**
         * @param numBins           The number of bins, including the most frequent one
         * @param numIndices        The number of examples not associated with the most frequent bin
         * @param sparseBinIndex    The index of the most frequent bin
         */
        AllocatedBinnedFeatureVector(uint32 numBins, uint32 numIndices, uint32 sparseBinIndex = 0)
            : BinnedFeatureVector(MemoryAllocator::allocateMemory<float32>(numBins - 1),
                                  MemoryAllocator::allocateMemory<uint32>(numIndices),
                                  MemoryAllocator::allocateMemory<uint32>(numBins + 1), numBins, numIndices,
                                  sparseBinIndex) {
            BinnedFeatureVector::indptr[0] = 0;
            BinnedFeatureVector::indptr[numBins] = numIndices;
        }

        /**
         * @param other A reference to an object of type `AllocatedBinnedFeatureVector` that should be copied
         */
        AllocatedBinnedFeatureVector(const AllocatedBinnedFeatureVector& other) : BinnedFeatureVector(other) {
            throw std::runtime_error("Objects of type AllocatedBinnedFeatureVector cannot be copied");
        }

        /**
         * @param other A reference to an object of type `AllocatedBinnedFeatureVector` that should be moved
         */
        AllocatedBinnedFeatureVector(AllocatedBinnedFeatureVector&& other) : BinnedFeatureVector(std::move(other)) {
            other.releaseThresholds();
            other.releaseIndices();
            other.releaseIndptr();
        }

        virtual ~AllocatedBinnedFeatureVector() override {
            MemoryAllocator::freeMemory(BinnedFeatureVector::thresholds);
            MemoryAllocator::freeMemory(BinnedFeatureVector::indices);
            MemoryAllocator::freeMemory(BinnedFeatureVector::indptr);
        }

        /**
         * Resizes the view by re-allocating the memory it provides access to.
         *
         * @param numValues     The number of bins, including the most frequent one
         * @param numIndices    The number of examples not associated with the most frequent bin
         */
        void resize(uint32 numBins, uint32 numIndices) {
            BinnedFeatureVector::thresholds =
              MemoryAllocator::reallocateMemory(BinnedFeatureVector::thresholds, numBins - 1);
            BinnedFeatureVector::indices = MemoryAllocator::reallocateMemory(BinnedFeatureVector::indices, numIndices);
            BinnedFeatureVector::indptr = MemoryAllocator::reallocateMemory(BinnedFeatureVector::indptr, numBins + 1);
            BinnedFeatureVector::numBins = numBins;
            BinnedFeatureVector::indptr[numBins] = numIndices;

            if (BinnedFeatureVector::sparseBinIndex >= numBins) {
                BinnedFeatureVector::sparseBinIndex = numBins - 1;
            }
        }
};
