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

}
