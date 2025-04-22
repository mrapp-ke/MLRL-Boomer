#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"

#include "mlrl/boosting/util/math.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseNonDecomposableStatisticView<StatisticType>::DenseNonDecomposableStatisticView(uint32 numRows, uint32 numCols)
        : CompositeMatrix<AllocatedCContiguousView<StatisticType>, AllocatedCContiguousView<StatisticType>>(
            AllocatedCContiguousView<StatisticType>(numRows, numCols),
            AllocatedCContiguousView<StatisticType>(numRows, util::triangularNumber(numCols)), numRows, numCols) {}

    template<typename StatisticType>
    DenseNonDecomposableStatisticView<StatisticType>::DenseNonDecomposableStatisticView(
      DenseNonDecomposableStatisticView<StatisticType>&& other)
        : CompositeMatrix<AllocatedCContiguousView<StatisticType>, AllocatedCContiguousView<StatisticType>>(
            std::move(other)) {}

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::gradient_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::gradients_cbegin(uint32 row) const {
        return this->firstView.values_cbegin(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::gradient_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::gradients_cend(uint32 row) const {
        return this->firstView.values_cend(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::gradient_iterator
      DenseNonDecomposableStatisticView<StatisticType>::gradients_begin(uint32 row) {
        return this->firstView.values_begin(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::gradient_iterator
      DenseNonDecomposableStatisticView<StatisticType>::gradients_end(uint32 row) {
        return this->firstView.values_end(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_cbegin(uint32 row) const {
        return this->secondView.values_cbegin(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_cend(uint32 row) const {
        return this->secondView.values_cend(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_begin(uint32 row) {
        return this->secondView.values_begin(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_end(uint32 row) {
        return this->secondView.values_end(row);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_diagonal_cbegin(uint32 row) const {
        return hessian_diagonal_const_iterator(this->secondView[row], 0);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticView<StatisticType>::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticView<StatisticType>::hessians_diagonal_cend(uint32 row) const {
        return hessian_diagonal_const_iterator(this->secondView[row], this->numCols);
    }

    template class DenseNonDecomposableStatisticView<float32>;
    template class DenseNonDecomposableStatisticView<float64>;
}
