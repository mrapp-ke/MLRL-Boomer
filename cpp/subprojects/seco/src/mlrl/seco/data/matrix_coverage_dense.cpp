#include "mlrl/seco/data/matrix_coverage_dense.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"

namespace seco {

    DenseCoverageMatrix::DenseCoverageMatrix(uint32 numRows, uint32 numCols, float64 sumOfUncoveredWeights)
        : DenseMatrixDecorator<AllocatedCContiguousView<uint32>>(
            AllocatedCContiguousView<uint32>(numRows, numCols, true)),
          sumOfUncoveredWeights_(sumOfUncoveredWeights) {}

    float64 DenseCoverageMatrix::getSumOfUncoveredWeights() const {
        return sumOfUncoveredWeights_;
    }

    void DenseCoverageMatrix::increaseCoverage(uint32 row, View<uint32>::const_iterator majorityLabelIndicesBegin,
                                               View<uint32>::const_iterator majorityLabelIndicesEnd,
                                               View<uint8>::const_iterator predictionBegin,
                                               View<uint8>::const_iterator predictionEnd,
                                               CompleteIndexVector::const_iterator indicesBegin,
                                               CompleteIndexVector::const_iterator indicesEnd) {
        uint32 numCols = this->getNumCols();
        value_iterator coverageIterator = this->values_begin(row);
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

        for (uint32 i = 0; i < numCols; i++) {
            bool predictedLabel = predictionBegin[i];
            bool majorityLabel = *majorityIterator;

            if (predictedLabel != majorityLabel) {
                uint32 coverage = coverageIterator[i];

                if (coverage == 0) {
                    sumOfUncoveredWeights_ -= 1;
                }

                coverageIterator[i] = coverage + 1;
            }

            majorityIterator++;
        }
    }

    void DenseCoverageMatrix::increaseCoverage(uint32 row, View<uint32>::const_iterator majorityLabelIndicesBegin,
                                               View<uint32>::const_iterator majorityLabelIndicesEnd,
                                               View<uint8>::const_iterator predictionBegin,
                                               View<uint8>::const_iterator predictionEnd,
                                               PartialIndexVector::const_iterator indicesBegin,
                                               PartialIndexVector::const_iterator indicesEnd) {
        uint32 numPredictions = indicesEnd - indicesBegin;
        value_iterator coverageIterator = this->values_begin(row);
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numPredictions; i++) {
            uint32 index = indicesBegin[i];
            bool predictedLabel = predictionBegin[i];
            std::advance(majorityIterator, index - previousIndex);
            bool majorityLabel = *majorityIterator;

            if (predictedLabel != majorityLabel) {
                uint32 coverage = coverageIterator[index];

                if (coverage == 0) {
                    sumOfUncoveredWeights_ -= 1;
                }

                coverageIterator[index] = coverage + 1;
            }

            previousIndex = index;
        }
    }

    void DenseCoverageMatrix::decreaseCoverage(uint32 row, View<uint32>::const_iterator majorityLabelIndicesBegin,
                                               View<uint32>::const_iterator majorityLabelIndicesEnd,
                                               View<uint8>::const_iterator predictionBegin,
                                               View<uint8>::const_iterator predictionEnd,
                                               CompleteIndexVector::const_iterator indicesBegin,
                                               CompleteIndexVector::const_iterator indicesEnd) {
        uint32 numCols = this->getNumCols();
        value_iterator coverageIterator = this->values_begin(row);
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

        for (uint32 i = 0; i < numCols; i++) {
            bool predictedLabel = predictionBegin[i];
            bool majorityLabel = *majorityIterator;

            if (predictedLabel != majorityLabel) {
                uint32 coverage = coverageIterator[i] - 1;

                if (coverage == 0) {
                    sumOfUncoveredWeights_ += 1;
                }

                coverageIterator[i] = coverage;
            }

            majorityIterator++;
        }
    }

    void DenseCoverageMatrix::decreaseCoverage(uint32 row, View<uint32>::const_iterator majorityLabelIndicesBegin,
                                               View<uint32>::const_iterator majorityLabelIndicesEnd,
                                               View<uint8>::const_iterator predictionBegin,
                                               View<uint8>::const_iterator predictionEnd,
                                               PartialIndexVector::const_iterator indicesBegin,
                                               PartialIndexVector::const_iterator indicesEnd) {
        uint32 numPredictions = indicesEnd - indicesBegin;
        value_iterator coverageIterator = this->values_begin(row);
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numPredictions; i++) {
            uint32 index = indicesBegin[i];
            bool predictedLabel = predictionBegin[i];
            std::advance(majorityIterator, index - previousIndex);
            bool majorityLabel = *majorityIterator;

            if (predictedLabel != majorityLabel) {
                uint32 coverage = coverageIterator[index] - 1;

                if (coverage == 0) {
                    sumOfUncoveredWeights_ += 1;
                }

                coverageIterator[index] = coverage;
            }

            previousIndex = index;
        }
    }

}
