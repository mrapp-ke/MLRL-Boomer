#include "seco/data/matrix_dense_weights.hpp"
#include "common/data/arrays.hpp"


namespace seco {

    DenseWeightMatrix::DenseWeightMatrix(uint32 numRows, uint32 numCols)
        : DenseMatrix<uint8>(numRows, numCols), sumOfUncoveredWeights_(0) {
        setArrayToValue<uint8>(this->array_, numRows * numCols, 1);
    }

    uint32 DenseWeightMatrix::getSumOfUncoveredWeights() const {
        return sumOfUncoveredWeights_;
    }

    void DenseWeightMatrix::setSumOfUncoveredWeights(uint32 sumOfUncoveredWeights) {
        sumOfUncoveredWeights_ = sumOfUncoveredWeights;
    }

    void DenseWeightMatrix::updateRow(uint32 row, const DenseVector<uint8>& majorityLabelVector,
                                      DenseVector<float64>::const_iterator predictionBegin,
                                      DenseVector<float64>::const_iterator predictionEnd,
                                      FullIndexVector::const_iterator indicesBegin,
                                      FullIndexVector::const_iterator indicesEnd) {
        uint32 numCols = this->getNumCols();
        iterator weightIterator = this->row_begin(row);
        DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVector.cbegin();

        for (uint32 i = 0; i < numCols; i++) {
            uint8 predictedLabel = (uint8) predictionBegin[i];
            uint8 majorityLabel = majorityIterator[i];

            if (predictedLabel != majorityLabel) {
                uint8 weight = weightIterator[i];

                if (weight) {
                    sumOfUncoveredWeights_ -= weight;
                    weightIterator[i] = 0;
                }
            }
        }
    }

    void DenseWeightMatrix::updateRow(uint32 row, const DenseVector<uint8>& majorityLabelVector,
                                      DenseVector<float64>::const_iterator predictionBegin,
                                      DenseVector<float64>::const_iterator predictionEnd,
                                      PartialIndexVector::const_iterator indicesBegin,
                                      PartialIndexVector::const_iterator indicesEnd) {
        uint32 numPredictions = indicesEnd - indicesBegin;
        iterator weightIterator = this->row_begin(row);
        DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVector.cbegin();

        for (uint32 i = 0; i < numPredictions; i++) {
            uint32 index = indicesBegin[i];
            uint8 predictedLabel = (uint8) predictionBegin[i];
            uint8 majorityLabel = majorityIterator[index];

            if (predictedLabel != majorityLabel) {
                uint8 weight = weightIterator[index];

                if (weight) {
                    sumOfUncoveredWeights_ -= weight;
                    weightIterator[index] = 0;
                }
            }
        }
    }

}
