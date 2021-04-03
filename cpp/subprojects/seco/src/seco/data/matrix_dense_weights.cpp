#include "seco/data/matrix_dense_weights.hpp"
#include "common/data/arrays.hpp"


namespace seco {

    template<class T>
    DenseWeightMatrix<T>::DenseWeightMatrix(uint32 numRows, uint32 numCols)
        : DenseMatrix<T>(numRows, numCols) {
        setArrayToValue<T>(this->array_, numRows * numCols, 1);
    }

    template class DenseWeightMatrix<uint8>;

}
