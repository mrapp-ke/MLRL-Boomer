/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/view_statistic_dense.hpp"
#include "mlrl/boosting/iterator/iterator_diagonal.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * non-decomposable loss function and are stored in pre-allocated C-contiguous arrays.
     *
     * @tparam StatisticType The type of the gradients and Hessians
     */
    template<typename StatisticType>
    class MLRLBOOSTING_API DenseNonDecomposableStatisticView : public DenseStatisticView<StatisticType> {
        public:

            /**
             * @param array         A pointer to an array of template type `T` that stores the gradients and Hessians
             * @param numRows       The number of rows in the view
             * @param numGradients  The number of gradients in each row of the view
             * @param numHessians   The number of Hessians in each row of the view
             * @param innerPadding  The number of unused elements to be inserted between gradients and Hessians
             * @param padding       The number of unused elements to be inserted at the end of each row
             */
            DenseNonDecomposableStatisticView(StatisticType* array, uint32 numRows, uint32 numGradients,
                                              uint32 numHessians, uint32 innerPadding = 0, uint32 padding = 0)
                : DenseStatisticView<StatisticType>(array, numRows, numGradients, numHessians, innerPadding, padding) {}

            /**
             * @param other A reference to an object of type `DenseNonDecomposableStatisticView` that should be copied
             */
            DenseNonDecomposableStatisticView(const DenseNonDecomposableStatisticView<StatisticType>& other)
                : DenseStatisticView<StatisticType>(other) {}

            /**
             * @param other A reference to an object of type `DenseNonDecomposableStatisticView` that should be moved
             */
            DenseNonDecomposableStatisticView(DenseNonDecomposableStatisticView<StatisticType>&& other)
                : DenseStatisticView<StatisticType>(std::move(other)) {}

            virtual ~DenseNonDecomposableStatisticView() override {}

            /**
             * An iterator that provides read-only access to the Hessians that correspond to the diagonal of the matrix.
             */
            using hessian_diagonal_const_iterator = DiagonalIterator<const StatisticType>;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the beginning of the Hessians that correspond to the
             * diagonal of the Hessian matrix at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_diagonal_const_iterator` to the beginning
             */
            hessian_diagonal_const_iterator hessians_diagonal_cbegin(uint32 row) const {
                return hessian_diagonal_const_iterator(View<const StatisticType>(this->hessians_cbegin(row)), 0);
            }

            /**
             * Returns a `hessian_diagonal_const_iterator` to the end of the Hessians that correspond to the diagonal of
             * the Hessian matrix at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_diagonal_const_iterator` to the end
             */
            hessian_diagonal_const_iterator hessians_diagonal_cend(uint32 row) const {
                return hessian_diagonal_const_iterator(View<const StatisticType>(this->hessians_cbegin(row)),
                                                       math::triangularNumber(this->numGradients_));
            }
    };

}
