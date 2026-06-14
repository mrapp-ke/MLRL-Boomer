/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/math/scalar_math.hpp"
#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that are stored in a single pre-allocated
     * C-contiguous array.
     *
     * @tparam StatisticType The type of the gradients and Hessians
     */
    template<typename StatisticType>
    class MLRLBOOSTING_API DenseStatisticView : public CContiguousView<StatisticType> {
        private:

            const uint32 numGradients_;

        public:

            /**
             * @param array         A pointer to an array of template type `T` that stores the gradients and Hessians
             * @param numRows       The number of rows in the view
             * @param numGradients  The number of gradients in each row of the view
             * @param numHessians   The number of Hessians in each row of the view
             */
            DenseStatisticView(StatisticType* array, uint32 numRows, uint32 numGradients, uint32 numHessians)
                : CContiguousView<StatisticType>(array, numRows, numGradients + numHessians),
                  numGradients_(numGradients) {}

            /**
             * @param other A reference to an object of type `DenseStatisticView` that should be copied
             */
            DenseStatisticView(const DenseStatisticView<StatisticType>& other)
                : CContiguousView<StatisticType>(other), numGradients_(other.numGradients_) {}

            /**
             * @param other A reference to an object of type `DenseStatisticView` that should be moved
             */
            DenseStatisticView(DenseStatisticView<StatisticType>&& other)
                : CContiguousView<StatisticType>(std::move(other)), numGradients_(other.numGradients_) {}

            virtual ~DenseStatisticView() override {}

            /**
             * An iterator that provides read-only access to the gradients.
             */
            using gradient_const_iterator = CContiguousView<StatisticType>::value_const_iterator;

            /**
             * An iterator that provides access to the gradients and allows to modify them.
             */
            using gradient_iterator = CContiguousView<StatisticType>::value_iterator;

            /**
             * An iterator that provides read-only access to the Hessians.
             */
            using hessian_const_iterator = CContiguousView<StatisticType>::value_const_iterator;

            /**
             * An iterator that provides access to the Hessians and allows to modify them.
             */
            using hessian_iterator = CContiguousView<StatisticType>::value_iterator;

            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the beginning of the given row
             */
            gradient_const_iterator gradients_cbegin(uint32 row) const {
                return this->values_cbegin(row);
            }

            /**
             * Returns a `gradient_const_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the end of the given row
             */
            gradient_const_iterator gradients_cend(uint32 row) const {
                return &(this->values_cbegin(row))[numGradients_];
            }

            /**
             * Returns a `gradient_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the beginning of the given row
             */
            gradient_iterator gradients_begin(uint32 row) {
                return this->values_begin(row);
            }

            /**
             * Returns a `gradient_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the end of the given row
             */
            gradient_iterator gradients_end(uint32 row) {
                return &(this->values_begin(row))[numGradients_];
            }

            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the beginning of the given row
             */
            hessian_const_iterator hessians_cbegin(uint32 row) const {
                return &(this->values_cbegin(row))[numGradients_];
            }

            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the end of the given row
             */
            hessian_const_iterator hessians_cend(uint32 row) const {
                return &(this->values_cbegin(row))[this->numCols];
            }

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the beginning of the given row
             */
            hessian_iterator hessians_begin(uint32 row) {
                return &(this->values_begin(row))[numGradients_];
            }

            /**
             * Returns a `hessian_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the end of the given row
             */
            hessian_iterator hessians_end(uint32 row) {
                return &(this->values_begin(row))[this->numCols];
            }

            /**
             * Returns the number of rows in the view.
             *
             * @return The number of rows
             */
            uint32 getNumRows() const {
                return this->numRows;
            }

            /**
             * Returns the number of gradients in each row of the view.
             *
             * @return The number of gradients in each row
             */
            uint32 getNumGradients() const {
                return numGradients_;
            }

            /**
             * Returns the number of Hessians in each row of the view.
             *
             * @return The number of Hessians in each row
             */
            uint32 getNumHessians() const {
                return math::triangularNumber(numGradients_);
            }
    };

    /**
     * Allocates the memory, a `DenseStatisticView` provides access to.
     *
     * @tparam View             The type of the view
     * @tparam MemoryAllocator  The type of the memory allocator to be used
     */
    template<typename View, typename MemoryAllocator = DefaultMemoryAllocator>
    class MLRLCOMMON_API DenseStatisticViewAllocator : public View {
        public:

            /**
             * @param numRows       The number of rows in the view
             * @param numGradients  The number of gradients in the view
             * @param numHessians   The number of Hessians in the view
             * @param init          True, if all elements in the view should be value-initialized, false otherwise
             */
            explicit DenseStatisticViewAllocator(uint32 numRows, uint32 numGradients, uint32 numHessians,
                                                 bool init = false)
                : View(MemoryAllocator::template allocateMemory<typename View::value_type>(
                         numRows * (numGradients + numHessians), init),
                       numRows, numGradients, numHessians) {}

            /**
             * @param other A reference to an object of type `DenseStatisticViewAllocator` that should be copied
             */
            DenseStatisticViewAllocator(const DenseStatisticViewAllocator<View, MemoryAllocator>& other) : View(other) {
                throw std::runtime_error("Objects of type DenseStatisticViewAllocator cannot be copied");
            }

            /**
             * @param other A reference to an object of type `DenseStatisticViewAllocator` that should be moved
             */
            DenseStatisticViewAllocator(DenseStatisticViewAllocator<View, MemoryAllocator>&& other)
                : View(std::move(other)) {
                other.release();
            }

            virtual ~DenseStatisticViewAllocator() override {
                MemoryAllocator::freeMemory(View::array);
            }
    };

}
