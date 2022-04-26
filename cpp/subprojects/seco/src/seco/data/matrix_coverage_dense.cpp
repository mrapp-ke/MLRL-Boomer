#include "seco/data/matrix_coverage_dense.hpp"
#include "common/iterator/binary_forward_iterator.hpp"


namespace seco {

    DenseCoverageMatrix::DenseCoverageMatrix(uint32 numRows, uint32 numCols, float64 sumOfUncoveredWeights)
        : DenseMatrix<uint32>(numRows, numCols, true), sumOfUncoveredWeights_(sumOfUncoveredWeights) {

    }

    float64 DenseCoverageMatrix::getSumOfUncoveredWeights() const {
        return sumOfUncoveredWeights_;
    }

    void DenseCoverageMatrix::increaseCoverage(uint32 row, const VectorConstView<uint32>& majorityLabelIndices,
                                               VectorView<float64>::const_iterator predictionBegin,
                                               VectorView<float64>::const_iterator predictionEnd,
                                               CompleteIndexVector::const_iterator indicesBegin,
                                               CompleteIndexVector::const_iterator indicesEnd) {
        uint32 numCols = this->getNumCols();
        value_iterator coverageIterator = this->row_values_begin(row);
        auto majorityIterator = make_binary_forward_iterator(majorityLabelIndices.cbegin(),
                                                             majorityLabelIndices.cend());

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

    void DenseCoverageMatrix::increaseCoverage(uint32 row, const VectorConstView<uint32>& majorityLabelIndices,
                                               VectorView<float64>::const_iterator predictionBegin,
                                               VectorView<float64>::const_iterator predictionEnd,
                                               PartialIndexVector::const_iterator indicesBegin,
                                               PartialIndexVector::const_iterator indicesEnd) {
        uint32 numPredictions = indicesEnd - indicesBegin;
        value_iterator coverageIterator = this->row_values_begin(row);
        auto majorityIterator = make_binary_forward_iterator(majorityLabelIndices.cbegin(),
                                                             majorityLabelIndices.cend());
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

}
