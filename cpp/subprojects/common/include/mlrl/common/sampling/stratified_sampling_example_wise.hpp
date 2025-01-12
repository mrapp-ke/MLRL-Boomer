/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/sampling/weight_vector_bit.hpp"

#include <memory>
#include <vector>

/**
 * Implements stratified sampling, where distinct label vectors are treated as individual classes.
 *
 * @tparam LabelMatrix      The type of the label matrix that provides random or row-wise access to the labels of the
 *                          training examples
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of the examples that should be
 *                          considered
 */
template<typename LabelMatrix, typename IndexIterator>
class ExampleWiseStratification final {
    private:

        const std::unique_ptr<RNG> rngPtr_;

        const uint32 numTotal_;

        std::vector<std::unique_ptr<std::vector<uint32>>> order_;

    public:

        /**
         * @param rngPtr        An unique pointer to an object of type `RNG` that should be used for generating random
         *                      numbers
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param indicesBegin  An iterator to the beginning of the indices of the examples that should be considered
         * @param indicesEnd    An iterator to the end of the indices of the examples that should be considered
         */
        ExampleWiseStratification(std::unique_ptr<RNG> rngPtr, const LabelMatrix& labelMatrix,
                                  IndexIterator indicesBegin, IndexIterator indicesEnd);

        /**
         * Randomly selects a stratified sample of the available examples and sets their weights to 1, while the
         * remaining weights are set to 0.
         *
         * @param weightVector  A reference to an object of type `BitWeightVector`, the weights should be written to
         * @param sampleSize    The fraction of the available examples to be selected
         * @param minSamples    The minimum number of examples to be included in the sample. Must be at least 1
         * @param maxSamples    The maximum number of examples to be included in the sample. Must be at least
         *                      `minSamples` or 0, if the number of examples should not be restricted
         */
        void sampleWeights(BitWeightVector& weightVector, float32 sampleSize, uint32 minSamples,
                           uint32 maxSamples) const;

        /**
         * Randomly splits the available examples into two distinct sets and updates a given `BiPartition` accordingly.
         *
         * @param partition A reference to an object of type `BiPartition` to be updated
         */
        void sampleBiPartition(BiPartition& partition) const;
};
