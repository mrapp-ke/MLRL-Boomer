/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/stratified_sampling_common.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>


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

        uint32 numTotal_;

        typedef typename LabelMatrix::view_type Key;

        typedef typename LabelMatrix::view_type::Hash Hash;

        typedef typename LabelMatrix::view_type::Pred Pred;

        std::unordered_map<Key, std::vector<uint32>, Hash, Pred> map_;

        std::vector<std::reference_wrapper<std::vector<uint32>>> order_;

    public:

        /**
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param indicesBegin  An iterator to the beginning of the indices of the examples that should be considered
         * @param indicesEnd    An iterator to the end of the indices of hte examples that should be considered
         */
        ExampleWiseStratification(const LabelMatrix& labelMatrix, IndexIterator indicesBegin,
                                  IndexIterator indicesEnd)
            : numTotal_(indicesEnd - indicesBegin) {
            // Create a map that stores the indices of the examples that are associated with each unique label vector...
            for (uint32 i = 0; i < numTotal_; i++) {
                uint32 exampleIndex = indicesBegin[i];
                std::vector<uint32>& exampleIndices = map_[labelMatrix.createView(exampleIndex)];
                exampleIndices.push_back(exampleIndex);
            }

            // Sort the label vectors by their frequency...
            order_.reserve(map_.size());

            for (auto it = map_.begin(); it != map_.end(); it++) {
                auto& entry = *it;
                std::vector<uint32>& exampleIndices = entry.second;
                order_.push_back(exampleIndices);
            }

            std::sort(order_.begin(), order_.end(), [=](const std::vector<uint32>& a, const std::vector<uint32>& b) {
                return a.size() < b.size();
            });
        }

        /**
         * Randomly selects a stratified sample of the available examples and sets their weights to 1, while the
         * remaining weights are set to 0.
         *
         * @param weightVector  A reference to an object of type `BitWeightVector`, the weights should be written to
         * @param sampleSize    The fraction of the available examples to be selected
         * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be
         *                      used
         */
        void sampleWeights(BitWeightVector& weightVector, float32 sampleSize, RNG& rng) const {
            uint32 numTotalSamples = (uint32) std::round(sampleSize * numTotal_);
            uint32 numTotalOutOfSamples = numTotal_ - numTotalSamples;
            uint32 numNonZeroWeights = 0;
            uint32 numZeroWeights = 0;

            for (auto it = order_.begin(); it != order_.end(); it++) {
                std::vector<uint32>& exampleIndices = *it;
                std::vector<uint32>::iterator indexIterator = exampleIndices.begin();
                uint32 numExamples = exampleIndices.size();
                float32 numSamplesDecimal = sampleSize * numExamples;
                uint32 numDesiredSamples = numTotalSamples - numNonZeroWeights;
                uint32 numDesiredOutOfSamples = numTotalOutOfSamples - numZeroWeights;
                uint32 numSamples = (uint32) (tiebreak(numDesiredSamples, numDesiredOutOfSamples, rng) ?
                                              std::ceil(numSamplesDecimal) : std::floor(numSamplesDecimal));
                numNonZeroWeights += numSamples;
                numZeroWeights += (numExamples - numSamples);

                // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and set their weight to 1...
                uint32 i;

                for (i = 0; i < numSamples; i++) {
                    uint32 randomIndex = rng.random(i, numExamples);
                    uint32 exampleIndex = indexIterator[randomIndex];
                    indexIterator[randomIndex] = indexIterator[i];
                    indexIterator[i] = exampleIndex;
                    weightVector.set(exampleIndex, true);
                }

                // Set the weights of the remaining examples to 0...
                for (; i < numExamples; i++) {
                    uint32 exampleIndex = indexIterator[i];
                    weightVector.set(exampleIndex, false);
                }
            }

            weightVector.setNumNonZeroWeights(numNonZeroWeights);
        }

        /**
         * Randomly splits the available examples into two distinct sets and updates a given `BiPartition` accordingly.
         *
         * @param partition A reference to an object of type `BiPartition` to be updated
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         */
        void sampleBiPartition(BiPartition& partition, RNG& rng) const {
            BiPartition::iterator firstIterator = partition.first_begin();
            BiPartition::iterator secondIterator = partition.second_begin();
            uint32 numFirst = partition.getNumFirst();
            uint32 numSecond = partition.getNumSecond();

            for (auto it = order_.begin(); it != order_.end(); it++) {
                std::vector<uint32>& exampleIndices = *it;
                std::vector<uint32>::iterator indexIterator = exampleIndices.begin();
                uint32 numExamples = exampleIndices.size();
                float32 sampleSize = (float32) numFirst / (float32) (numFirst + numSecond);
                float32 numSamplesDecimal = sampleSize * numExamples;
                uint32 numSamples = (uint32) (tiebreak(numFirst, numSecond, rng) ? std::ceil(numSamplesDecimal)
                                                                                 : std::floor(numSamplesDecimal));
                numFirst -= numSamples;
                numSecond -= (numExamples - numSamples);

                // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and add them to the first set...
                uint32 i;

                for (i = 0; i < numSamples; i++) {
                    uint32 randomIndex = rng.random(i, numExamples);
                    uint32 exampleIndex = indexIterator[randomIndex];
                    indexIterator[randomIndex] = indexIterator[i];
                    indexIterator[i] = exampleIndex;
                    *firstIterator = exampleIndex;
                    firstIterator++;
                }

                // Add the remaining examples to the second set...
                for (; i < numExamples; i++) {
                    uint32 exampleIndex = indexIterator[i];
                    *secondIterator = exampleIndex;
                    secondIterator++;
                }
            }
        }

};
