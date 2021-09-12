/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/matrix_lil.hpp"


namespace boosting {

    /**
     * A two-dimensional matrix that provides row-wise access to values that are stored in the list of lists (LIL)
     * format.
     *
     * @tparam T The type of the values that are stored in the matrix
     */
    template<typename T>
    class NumericLilMatrix final : public LilMatrix<T> {

        public:

            /**
             * @param numRows The number of rows in the matrix
             */
            NumericLilMatrix(uint32 numRows);

    };

}
