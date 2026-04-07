#include "mlrl/boosting/data/view_statistic_decomposable_dense.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseDecomposableStatisticView<StatisticType>::DenseDecomposableStatisticView(uint32 numRows, uint32 numCols)
        : CompositeMatrix<AllocatedCContiguousView<StatisticType>, AllocatedCContiguousView<StatisticType>>(
            AllocatedCContiguousView<StatisticType>(numRows, numCols),
            AllocatedCContiguousView<StatisticType>(numRows, numCols), numRows, numCols) {}

    template<typename StatisticType>
    DenseDecomposableStatisticView<StatisticType>::DenseDecomposableStatisticView(
      DenseDecomposableStatisticView<StatisticType>&& other)
        : CompositeMatrix<AllocatedCContiguousView<StatisticType>, AllocatedCContiguousView<StatisticType>>(
            std::move(other)) {}

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::gradient_const_iterator
      DenseDecomposableStatisticView<StatisticType>::gradients_cbegin(uint32 row) const {
        return this->firstView.values_cbegin(row);
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::gradient_const_iterator
      DenseDecomposableStatisticView<StatisticType>::gradients_cend(uint32 row) const {
        return this->firstView.values_cend(row);
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::gradient_iterator
      DenseDecomposableStatisticView<StatisticType>::gradients_begin(uint32 row) {
        return this->firstView.values_begin(row);
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::gradient_iterator
      DenseDecomposableStatisticView<StatisticType>::gradients_end(uint32 row) {
        return this->firstView.values_end(row);
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::hessian_const_iterator
      DenseDecomposableStatisticView<StatisticType>::hessians_cbegin(uint32 row) const {
        return this->secondView.values_cbegin(row);
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::hessian_const_iterator
      DenseDecomposableStatisticView<StatisticType>::hessians_cend(uint32 row) const {
        return this->secondView.values_cend(row);
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::hessian_iterator
      DenseDecomposableStatisticView<StatisticType>::hessians_begin(uint32 row) {
        return this->secondView.values_begin(row);
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::hessian_iterator
      DenseDecomposableStatisticView<StatisticType>::hessians_end(uint32 row) {
        return this->secondView.values_end(row);
    }

    template class DenseDecomposableStatisticView<float32>;
    template class DenseDecomposableStatisticView<float64>;
}
