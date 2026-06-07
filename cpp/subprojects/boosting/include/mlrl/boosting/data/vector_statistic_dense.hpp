/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/data/view_vector.hpp"

namespace boosting {

    /**
     * A one-dimensional view that provides access to gradients and Hessians that are stored in a single pre-allocated
     * array.
     *
     * @tparam StatisticType The type of the gradient and Hessians
     */
    template<typename StatisticType>
    class MLRLBOOSTING_API DenseStatisticVectorView : public Vector<StatisticType> {
        private:

            const uint32 numGradients_;

        public:

            /**
             * @param array         A pointer to an array of template type `StatisticType` that stores the gradients and
             *                      Hessians
             * @param numGradients  The number of gradients in the view
             * @param numHessians   The number of Hessians in the view
             */
            DenseStatisticVectorView(StatisticType* array, uint32 numGradients, uint32 numHessians)
                : Vector<StatisticType>(array, numGradients + numHessians), numGradients_(numGradients) {}

            /**
             * @param other A reference to an object of type `DenseStatisticVectorView` that should be copied
             */
            DenseStatisticVectorView(const DenseStatisticVectorView<StatisticType>& other)
                : Vector<StatisticType>(other), numGradients_(other.numGradients_) {}

            /**
             * @param other A reference to an object of type `DenseStatisticVectorView` that should be moved
             */
            DenseStatisticVectorView(DenseStatisticVectorView<StatisticType>&& other)
                : Vector<StatisticType>(std::move(other)), numGradients_(other.numGradients_) {}

            virtual ~DenseStatisticVectorView() override {}

            /**
             * The type of the gradients and Hessians.
             */
            using statistic_type = StatisticType;

            /**
             * An iterator that provides access to the gradients in the view and allows to modify them.
             */
            using gradient_iterator = View<StatisticType>::iterator;

            /**
             * An iterator that provides read-only access to the gradients in the view.
             */
            using gradient_const_iterator = View<StatisticType>::const_iterator;

            /**
             * An iterator that provides access to the Hessians in the view and allows to modify them.
             */
            using hessian_iterator = View<StatisticType>::iterator;

            /**
             * An iterator that provides read-only access to the Hessians in the view.
             */
            using hessian_const_iterator = View<StatisticType>::const_iterator;

            /**
             * Returns a `gradient_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_iterator` to the beginning
             */
            gradient_iterator gradients_begin() {
                return this->begin();
            }

            /**
             * Returns a `gradient_iterator` to the end of the gradients.
             *
             * @return A `gradient_iterator` to the end
             */
            gradient_iterator gradients_end() {
                return &(this->begin())[numGradients_];
            }

            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_const_iterator` to the beginning
             */
            gradient_const_iterator gradients_cbegin() const {
                return this->cbegin();
            }

            /**
             * Returns a `gradient_const_iterator` to the end of the gradients.
             *
             * @return A `gradient_const_iterator` to the end
             */
            gradient_const_iterator gradients_cend() const {
                return &(this->cbegin())[numGradients_];
            }

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_iterator` to the beginning
             */
            hessian_iterator hessians_begin() {
                return &(this->begin())[numGradients_];
            }

            /**
             * Returns a `hessian_iterator` to the end of the Hessians.
             *
             * @return A `hessian_iterator` to the end
             */
            hessian_iterator hessians_end() {
                return &(this->begin())[this->numElements];
            }

            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_const_iterator` to the beginning
             */
            hessian_const_iterator hessians_cbegin() const {
                return &(this->cbegin())[numGradients_];
            }

            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians.
             *
             * @return A `hessian_const_iterator` to the end
             */
            hessian_const_iterator hessians_cend() const {
                return &(this->cbegin())[this->numElements];
            }

            /**
             * Returns the number of gradients in the view.
             *
             * @return The number of gradients
             */
            uint32 getNumGradients() const {
                return numGradients_;
            }

            /**
             * Returns the number of Hessians in the view.
             *
             + @return The number of Hessians
             */
            uint32 getNumHessians() const {
                return this->numElements - numGradients_;
            }
    };
}

/**
 * Allocates the memory, a `DenseStatisticVectorView` provides access to.
 *
 * @tparam View The type of the view
 */
template<typename View>
class MLRLCOMMON_API DenseStatisticVectorAllocator : public View {
    public:

        /**
         * @param numGradients  The number of elements in the view
         * @param numHessians   The number of Hessians in the view
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit DenseStatisticVectorAllocator(uint32 numGradients, uint32 numHessians, bool init = false)
            : View(MemoryAllocator::allocateMemory<typename View::value_type>(numGradients + numHessians, init),
                   numGradients, numHessians) {}

        /**
         * @param other A reference to an object of type `DenseStatisticVectorAllocator` that should be copied
         */
        DenseStatisticVectorAllocator(const DenseStatisticVectorAllocator<View>& other) : View(other) {
            throw std::runtime_error("Objects of type DenseStatisticVectorAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `DenseStatisticVectorAllocator` that should be moved
         */
        DenseStatisticVectorAllocator(DenseStatisticVectorAllocator<View>&& other) : View(std::move(other)) {
            other.release();
        }

        virtual ~DenseStatisticVectorAllocator() override {
            MemoryAllocator::freeMemory(View::array);
        }

        /**
         * The type of the view for which this allocated manages memory.
         */
        using allocated_view_type = View;

        /**
         * Returns a const reference to the view for which this allocator manages memory.
         *
         * @return A const reference to an object of template type `View` for which this allocator manages memory
         */
        const View& getAllocatedView() const {
            return *this;
        }

        /**
         * Returns a reference to the view for which this allocator manages memory.
         *
         * @return A reference to an object of template type `View` for which this allocator manages memory
         */
        View& getAllocatedView() {
            return *this;
        }
};
