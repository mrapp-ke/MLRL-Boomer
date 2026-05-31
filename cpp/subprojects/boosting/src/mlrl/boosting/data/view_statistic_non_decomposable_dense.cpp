#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"

#include "mlrl/boosting/math/scalar_math.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseNonDecomposableStatisticView<StatisticType>::DenseNonDecomposableStatisticView(uint32 numRows, uint32 numCols)
        : AllocatedCContiguousView<StatisticType>(numRows, numCols + math::triangularNumber(numCols)),
          numGradients_(numCols) {}

    template<typename StatisticType>
    DenseNonDecomposableStatisticView<StatisticType>::DenseNonDecomposableStatisticView(
      DenseNonDecomposableStatisticView<StatisticType>&& other)
        : AllocatedCContiguousView<StatisticType>(std::move(other)), numGradients_(other.numGradients_) {}

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::gradient_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::gradients_cbegin(uint32 row) const {
        return this->values_cbegin(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::gradient_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::gradients_cend(uint32 row) const {
        return &(this->values_cbegin(row))[numGradients_];
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::gradient_iterator
      DenseNonDecomposableStatisticView<StatisticType>::gradients_begin(uint32 row) {
        return this->values_begin(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::gradient_iterator
      DenseNonDecomposableStatisticView<StatisticType>::gradients_end(uint32 row) {
        return &(this->values_begin(row))[numGradients_];
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_cbegin(uint32 row) const {
        return &(this->values_cbegin(row))[numGradients_];
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_cend(uint32 row) const {
        return &(this->values_cbegin(row))[this->numCols];
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_begin(uint32 row) {
        return &(this->values_begin(row))[numGradients_];
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_end(uint32 row) {
        return &(this->values_begin(row))[this->numCols];
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_diagonal_cbegin(uint32 row) const {
        return hessian_diagonal_const_iterator(View<const StatisticType>(this->hessians_cbegin(row)), 0);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_diagonal_cend(uint32 row) const {
        return hessian_diagonal_const_iterator(View<const StatisticType>(this->hessians_cbegin(row)),
                                               math::triangularNumber(numGradients_));
    }

    template<typename StatisticType>
    uint32 DenseNonDecomposableStatisticView<StatisticType>::getNumRows() const {
        return this->numRows;
    }

    template<typename StatisticType>
    uint32 DenseNonDecomposableStatisticView<StatisticType>::getNumCols() const {
        return numGradients_;
    }

    template class DenseNonDecomposableStatisticView<float32>;
    template class DenseNonDecomposableStatisticView<float64>;
}
