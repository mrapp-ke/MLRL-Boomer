#include "boosting/data/matrix_lil_numeric.hpp"


namespace boosting {

    template<typename T>
    NumericLilMatrix<T>::NumericLilMatrix(uint32 numRows)
        : LilMatrix<T>(numRows) {

    }

    template class NumericLilMatrix<uint8>;
    template class NumericLilMatrix<uint32>;
    template class NumericLilMatrix<float32>;
    template class NumericLilMatrix<float64>;

}
