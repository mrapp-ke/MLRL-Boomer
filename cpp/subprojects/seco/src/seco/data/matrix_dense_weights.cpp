#include "seco/data/matrix_dense_weights.hpp"
#include "common/data/arrays.hpp"


namespace seco {

    template<class T>
    DenseWeightMatrix<T>::DenseWeightMatrix(uint32 numRows, uint32 numCols)
        : DenseMatrix<T>(numRows, numCols) {
        setArrayToValue(DenseMatrix<T>::array_, numRows * numCols, 1);
    }

}
