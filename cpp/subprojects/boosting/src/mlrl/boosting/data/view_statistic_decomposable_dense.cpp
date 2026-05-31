#include "mlrl/boosting/data/view_statistic_decomposable_dense.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseDecomposableStatisticView<StatisticType>::DenseDecomposableStatisticView(uint32 numRows, uint32 numCols)
        : AllocatedCContiguousView<StatisticType>(numRows, numCols * 2) {}

    template<typename StatisticType>
    DenseDecomposableStatisticView<StatisticType>::DenseDecomposableStatisticView(
      DenseDecomposableStatisticView<StatisticType>&& other)
        : AllocatedCContiguousView<StatisticType>(std::move(other)) {}

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::gradient_const_iterator
      DenseDecomposableStatisticView<StatisticType>::gradients_cbegin(uint32 row) const {
        return this->values_cbegin(row);
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::gradient_const_iterator
      DenseDecomposableStatisticView<StatisticType>::gradients_cend(uint32 row) const {
        return &(this->values_cbegin(row))[this->getNumCols()];
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::gradient_iterator
      DenseDecomposableStatisticView<StatisticType>::gradients_begin(uint32 row) {
        return this->values_begin(row);
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::gradient_iterator
      DenseDecomposableStatisticView<StatisticType>::gradients_end(uint32 row) {
        return &(this->values_begin(row))[this->getNumCols()];
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::hessian_const_iterator
      DenseDecomposableStatisticView<StatisticType>::hessians_cbegin(uint32 row) const {
        return &(this->values_cbegin(row))[this->getNumCols()];
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::hessian_const_iterator
      DenseDecomposableStatisticView<StatisticType>::hessians_cend(uint32 row) const {
        return &(this->values_cbegin(row))[this->numCols];
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::hessian_iterator
      DenseDecomposableStatisticView<StatisticType>::hessians_begin(uint32 row) {
        return &(this->values_begin(row))[this->getNumCols()];
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticView<StatisticType>::hessian_iterator
      DenseDecomposableStatisticView<StatisticType>::hessians_end(uint32 row) {
        return &(this->values_begin(row))[this->numCols];
    }

    template<typename StatisticType>
    uint32 DenseDecomposableStatisticView<StatisticType>::getNumRows() const {
        return this->numRows;
    }

    template<typename StatisticType>
    uint32 DenseDecomposableStatisticView<StatisticType>::getNumCols() const {
        return this->numCols / 2;
    }

    template class DenseDecomposableStatisticView<float32>;
    template class DenseDecomposableStatisticView<float64>;
}
