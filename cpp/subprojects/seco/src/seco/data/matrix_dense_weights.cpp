#include "seco/data/matrix_dense_weights.hpp"
#include "common/data/arrays.hpp"


namespace seco {

    DenseWeightMatrix::DenseWeightMatrix(uint32 numRows, uint32 numCols)
        : DenseMatrix<uint8>(numRows, numCols) {
        setArrayToValue<uint8>(this->array_, numRows * numCols, 1);
    }

}
