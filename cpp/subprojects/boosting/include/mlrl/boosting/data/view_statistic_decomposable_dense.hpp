/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/statistic.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * decomposable loss function and are stored in pre-allocated C-contiguous arrays.
     *
     * @tparam StatisticType The type of the gradients and Hessians
     */
    template<typename StatisticType>
    using DenseDecomposableStatisticView = CContiguousView<Statistic<StatisticType>>;
}
