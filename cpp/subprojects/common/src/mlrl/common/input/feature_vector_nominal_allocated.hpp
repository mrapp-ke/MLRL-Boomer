/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_nominal.hpp"

#include <utility>

/**
 * Allocates the memory, a `NominalFeatureVector` provides access to.
 */
class MLRLCOMMON_API AllocatedNominalFeatureVector : public NominalFeatureVector {
    public:

        /**
         * @param numValues     The number of nominal values
         * @param numIndices    The number of examples not associated with the majority value
         * @param majorityValue The majority value, i.e., the most frequent value, of the nominal feature
         */
        AllocatedNominalFeatureVector(uint32 numValues, uint32 numIndices, int32 majorityValue = 0)
            : NominalFeatureVector(util::allocateMemory<int32>(numValues), util::allocateMemory<uint32>(numIndices),
                                   util::allocateMemory<uint32>(numValues + 1), numValues, numIndices, majorityValue) {
            NominalFeatureVector::indptr[0] = 0;
            NominalFeatureVector::indptr[numValues] = numIndices;
        }

        /**
         * @param other A reference to an object of type `AllocatedNominalFeatureVector` that should be copied
         */
        AllocatedNominalFeatureVector(const AllocatedNominalFeatureVector& other) : NominalFeatureVector(other) {
            throw std::runtime_error("Objects of type AllocatedNominalFeatureVector cannot be copied");
        }

        /**
         * @param other A reference to an object of type `AllocatedNominalFeatureVector` that should be moved
         */
        AllocatedNominalFeatureVector(AllocatedNominalFeatureVector&& other) : NominalFeatureVector(std::move(other)) {
            other.releaseValues();
            other.releaseIndices();
            other.releaseIndptr();
        }

        virtual ~AllocatedNominalFeatureVector() override {
            util::freeMemory(NominalFeatureVector::values);
            util::freeMemory(NominalFeatureVector::indices);
            util::freeMemory(NominalFeatureVector::indptr);
        }

        /**
         * Resizes the view by re-allocating the memory it provides access to.
         *
         * @param numValues     The number of nominal values
         * @param numIndices    The number of examples not associated with the majority value to which the view should
         *                      be resized
         */
        void resize(uint32 numValues, uint32 numIndices) {
            NominalFeatureVector::values = util::reallocateMemory(NominalFeatureVector::values, numValues);
            NominalFeatureVector::indices = util::reallocateMemory(NominalFeatureVector::indices, numIndices);
            NominalFeatureVector::indptr = util::reallocateMemory(NominalFeatureVector::indptr, numValues + 1);
            NominalFeatureVector::numBins = numValues;
            NominalFeatureVector::indptr[numValues] = numIndices;
        }
};
