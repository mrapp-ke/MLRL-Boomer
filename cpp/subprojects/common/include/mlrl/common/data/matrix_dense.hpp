/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_dense.hpp"

/**
 * Provides row-wise read and write access via iterators to the values stored in a dense matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
using DenseMatrixDecorator = IterableDenseMatrixDecorator<MatrixDecorator<Matrix>>;
