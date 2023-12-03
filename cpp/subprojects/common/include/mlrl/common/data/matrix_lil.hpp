/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/list_of_lists.hpp"
#include "mlrl/common/data/view_matrix_lil.hpp"

/**
 * Provides row-wise read and write access via iterators to the values stored in a sparse matrix in the list of lists
 * (LIL) format.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
using LilMatrixDecorator = IterableListOfListsDecorator<MatrixDecorator<Matrix>>;

/**
 * A two-dimensional matrix that provides row-wise access to data that is stored in the list of lists (LIL) format.
 *
 * @tparam T The type of the data that is stored by the matrix
 */
template<typename T>
using LilMatrix = ListOfLists<IndexedValue<T>>;
