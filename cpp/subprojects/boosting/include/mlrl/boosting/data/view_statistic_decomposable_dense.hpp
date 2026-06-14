/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/view_statistic_dense.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * decomposable loss function and are stored in a single pre-allocated C-contiguous array.
     *
     * @tparam StatisticType The type of the gradients and Hessians
     */
    template<typename StatisticType>
    using DenseDecomposableStatisticView = DenseStatisticView<StatisticType>;
}
